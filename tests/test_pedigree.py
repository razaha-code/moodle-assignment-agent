import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.llm import LLMAnswerGenerator
from src.llm_providers import LLMResponse
from src.models import Config, DocumentManifest, ManifestQuestion
from src.pedigree import parse_pedigrees, pedigree_html, requires_pedigree
from src.output import build_rich_html_document


FIXTURE = Path(__file__).parent / "fixtures" / "genetics_pedigree.json"


class PedigreeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.data = self.fixture["pedigree"]
        self.source = self.fixture["source"]

    def test_source_family_is_eight_people_three_affected_and_two_male_siblings(self):
        pedigree = parse_pedigrees([self.data], self.source)[0]
        self.assertEqual(8, len(pedigree.people))
        self.assertEqual(3, sum(p["affected"] for p in pedigree.people))
        by_id = {p["id"]: p for p in pedigree.people}
        for person in ("hb", "wb"):
            self.assertEqual("M", by_id[person]["sex"])
            self.assertTrue(by_id[person]["affected"])
        self.assertEqual(3, len(pedigree.partnerships))
        self.assertTrue(requires_pedigree(self.source))
        self.assertFalse(requires_pedigree("Explain the inheritance pattern in this pedigree"))

    def test_rejects_duplicate_missing_parent_cycles_and_unquoted_facts(self):
        mutations = [
            lambda d: d["people"].append(copy.deepcopy(d["people"][0])),
            lambda d: d["people"][-1].update(father="missing"),
            lambda d: d["people"][0].update(father="h", evidence={**d["people"][0]["evidence"], "father":self.source}),
            lambda d: d["people"][0]["evidence"].update(sex="invented quote"),
            lambda d: d["people"][0].update(affected="false"),
            lambda d: d.update(partnerships=[]),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises(ValueError):
                    parse_pedigrees([data], self.source)

    def test_unknown_status_stays_unknown_and_labels_are_escaped(self):
        self.data["people"][0].update(affected=None, carrier=None)
        self.data["title"] = '<script>alert("x")</script>'
        p = parse_pedigrees([self.data], self.source)[0]
        self.assertIsNone(p.people[0]["affected"])
        figure = pedigree_html(p)
        self.assertNotIn("<script>", figure)
        self.assertIn("&lt;script&gt;", figure)
        document = build_rich_html_document("Test", figure)
        self.assertIn("pedigreejs v5.0.2", document)
        self.assertNotIn('src="https://cdn.jsdelivr.net/npm/d3', document)

    def generator(self):
        config = Config("https://example.com", "token", "service", "key", "model", "Asia/Jerusalem", set())
        provider = Mock()
        registry = Mock()
        registry.get.return_value = provider
        generator = LLMAnswerGenerator(config, registry)
        generator.refine_low_confidence_manifest_questions = Mock()
        manifest = DocumentManifest(3, "hash", "model", 1, "genetics", [
            ManifestQuestion("1a", "a", "1", (1,), self.source, "Draw a pedigree"),
            ManifestQuestion("2", "2", "2", (1,), "Explain a gene", "Explain a gene"),
        ], "now")
        return generator, provider, manifest

    def run_generation(self, generator, manifest, root):
        return generator.generate_specialist_document_from_manifest(
            SimpleNamespace(assignment_id=1), manifest, "solve", "teacher", "Solve",
            checkpoint_path=root / "checkpoint.json", manifest_path=root / "manifest.json", page_image_paths=[])

    def response(self, diagrams=None, item="1a"):
        answer = {"id": item, "html": "<p>Explanation of the family relationships.</p>"}
        if diagrams is not None:
            answer["pedigrees"] = diagrams
        return LLMResponse(json.dumps({"answers": [answer]}), "stop")

    def test_source_review_rejects_wrong_sex_then_retries_and_resumes(self):
        generator, provider, manifest = self.generator()
        wrong = copy.deepcopy(self.data)
        wrong["people"][-1]["sex"] = "U"
        provider.generate.side_effect = [
            self.response([wrong]), LLMResponse('{"errors":["Brother must be male"]}', "stop"),
            self.response(item="2"), self.response([self.data]), LLMResponse('{"errors":[]}', "stop"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_generation(generator, manifest, root)
            self.assertTrue(result.complete)
            self.assertEqual(2, result.attempts["1a"])
            self.assertIn("Brother must be male", provider.generate.call_args_list[3].kwargs["user_content"][0]["text"])
            resumed = self.run_generation(generator, manifest, root)
            self.assertTrue(resumed.complete)
            self.assertEqual(result.answers[0].pedigrees, resumed.answers[0].pedigrees)
            self.assertEqual(5, provider.generate.call_count)

    def test_missing_pedigree_is_incomplete_after_bounded_attempts(self):
        generator, provider, manifest = self.generator()
        provider.generate.side_effect = [self.response(), self.response(item="2"), self.response(), self.response()]
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_generation(generator, manifest, Path(directory))
        self.assertEqual(["1a"], result.missing_ids)
        self.assertEqual(3, result.attempts["1a"])

    def test_legacy_checkpoint_repairs_only_required_diagram(self):
        generator, provider, manifest = self.generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = {"source_hash": "hash", "manifest_fingerprint": generator._manifest_fingerprint(manifest),
                   "model": "model", "mode": "solve", "specialist_name": "teacher", "schema_version": 1,
                   "attempts": {"1a":3,"2":1}, "answers": [
                       {"id":"1a","html":"<p>Old ASCII family tree.</p>"},
                       {"id":"2","html":"<p>Existing gene explanation.</p>"}]}
            (root / "checkpoint.json").write_text(json.dumps(old), encoding="utf-8")
            provider.generate.side_effect = [self.response([self.data]), LLMResponse('{"errors":[]}', "stop")]
            result = self.run_generation(generator, manifest, root)
            self.assertTrue(result.complete)
            self.assertEqual(1, result.attempts["1a"])
            self.assertEqual("<p>Existing gene explanation.</p>", result.answers[1].html)
            prompt = provider.generate.call_args_list[0].kwargs["user_content"][0]["text"]
            self.assertIn('"id": "2", "answer_requested": false', prompt)


@unittest.skipUnless(os.environ.get("RUN_PEDIGREE_BROWSER_TESTS") == "1", "Enable offline Chromium tests with RUN_PEDIGREE_BROWSER_TESTS=1")
class PedigreeBrowserTests(unittest.TestCase):
    def test_complex_family_half_siblings_multiple_partners_and_cousins(self):
        from playwright.sync_api import sync_playwright
        people = []
        rows = [('gf','M',None,None), ('gm','F',None,None),
                ('a','F','gf','gm'), ('b','M','gf','gm'),
                ('c','M',None,None), ('d','F',None,None), ('c2','M',None,None),
                ('e','M','c','a'), ('f','F','b','d'), ('h','F','c2','a'), ('g','F','e','f')]
        for id, sex, father, mother in rows:
            people.append(dict(id=id,label=id,sex=sex,affected=False,carrier=None,father=father,mother=mother,
                               evidence={k:'given' for k in ('label','sex','affected','father','mother')}))
        pairs = [dict(partners=[a,b],consanguineous=related,evidence='given') for a,b,related in
                 [('gf','gm',False),('c','a',False),('b','d',False),('c2','a',False),('e','f',True)]]
        data = dict(title='Complex family',people=people,partnerships=pairs)
        pedigree = parse_pedigrees([data], 'given')[0]
        document = build_rich_html_document('Complex family',pedigree_html(pedigree))
        with sync_playwright() as p:
            browser=p.chromium.launch()
            page=browser.new_page()
            page.route('https://**/*',lambda route:route.abort())
            page.set_content(document)
            page.wait_for_function('window.__pedigreesReady || window.__pedigreeError')
            self.assertIsNone(page.evaluate('window.__pedigreeError'))
            self.assertEqual(11,page.locator('[data-person-id]').count())
            self.assertEqual(2,page.locator('[data-partners="e f"]').count())
            self.assertEqual(1,page.locator('[data-partners="c2 a"]').count())
            browser.close()

    def test_offline_family_symbols_relationships_and_multiple_diagrams(self):
        from playwright.sync_api import sync_playwright
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        pedigree = parse_pedigrees([fixture["pedigree"]], fixture["source"])[0]
        body = pedigree_html(pedigree) * 2
        document = build_rich_html_document("Offline pedigrees", body, output_style="ipad_clean")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.route("http://**/*", lambda route: route.abort())
            page.route("https://**/*", lambda route: route.abort())
            page.set_content(document)
            page.wait_for_function("window.__pedigreesReady || window.__pedigreeError")
            self.assertIsNone(page.evaluate("window.__pedigreeError"))
            self.assertEqual(16, page.locator("[data-person-id]").count())
            self.assertEqual(6, page.locator('[data-affected="true"]').count())
            self.assertEqual(2, page.locator('[data-partners="h w"]').count())
            self.assertEqual(2, page.locator('[data-person-id="hb"][data-sex="M"]').count())
            self.assertEqual(16, page.evaluate("new Set(Array.from(document.querySelectorAll('clipPath'), el => el.id)).size"))
            self.assertEqual("rtl", page.locator('[data-person-id="h"] text').first.get_attribute("direction"))
            browser.close()
