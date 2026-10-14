import unittest

from src.retriever import get_recruiter_tools


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def similarity_search(self, **kwargs):
        self.calls.append(kwargs)
        return []


class RetrieverTests(unittest.TestCase):
    def test_empty_filtered_search_does_not_search_other_document_types(self):
        for tool_name, doc_type in (
            ("query_job_requirements", "job_description"),
            ("query_company_culture", "culture"),
        ):
            with self.subTest(doc_type=doc_type):
                vector_store = FakeVectorStore()
                tools = get_recruiter_tools(vector_store)
                tool = next(item for item in tools if item.name == tool_name)

                result = tool.run("teamwork")

                self.assertEqual(len(vector_store.calls), 1)
                self.assertEqual(
                    vector_store.calls[0]["filter"], {"doc_type": doc_type}
                )
                self.assertIn("No relevant context", result)


if __name__ == "__main__":
    unittest.main()