import html
import re
from typing import Any, Dict, Optional
from bs4 import BeautifulSoup
import httpx
import trafilatura


class DataExtractor:
    """3-Layer Fallback Extractor:

    1. Direct ATS API (Greenhouse / Lever) -> Structured JSON
    2. Jina Reader API -> Clean Markdown
    3. Trafilatura -> Local HTML stripping
    """

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "RecruiterSimulator/1.0 (Educational RAG Project)"
        }

    def _extract_greenhouse(self, url: str) -> Optional[Dict[str, Any]]:
        pattern = r"greenhouse\.io/(?:embed/job_board/)?([^/]+)/jobs/(\d+)"
        match = re.search(pattern, url)
        if not match:
            return None

        board_token, job_id = match.group(1), match.group(2)
        api_url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs/{job_id}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(api_url)
                if res.status_code == 200:
                    data = res.json()
                    raw_content = html.unescape(data.get("content", ""))
                    soup = BeautifulSoup(raw_content, "html.parser")
                    clean_text = soup.get_text(separator="\n\n").strip()

                    markdown_text = (
                        f"# Job Title: {data.get('title')}\n"
                        f"**Company/Board:** {board_token}\n"
                        f"**Location:** {data.get('location', {}).get('name', 'N/A')}\n"
                        f"**Updated At:** {data.get('updated_at', 'N/A')}\n\n"
                        f"## Job Description\n{clean_text}"
                    )
                    return {
                        "text": markdown_text,
                        "strategy": "greenhouse_api",
                        "title": data.get("title", "Job Posting"),
                    }
        except Exception as e:
            print(f"[Warning] Greenhouse API failed: {e}. Falling back to Jina.")
        return None

    def _extract_lever(self, url: str) -> Optional[Dict[str, Any]]:
        pattern = r"jobs\.lever\.co/([^/]+)/([a-f0-9\-]+)"
        match = re.search(pattern, url)
        if not match:
            return None

        site, posting_id = match.group(1), match.group(2)
        api_url = f"https://api.lever.co/v0/postings/{site}/{posting_id}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(api_url)
                if res.status_code == 200:
                    data = res.json()
                    desc_soup = BeautifulSoup(
                        data.get("descriptionHtml", ""), "html.parser"
                    )
                    description = desc_soup.get_text(separator="\n\n").strip()

                    lists_text = ""
                    for lst in data.get("lists", []):
                        list_soup = BeautifulSoup(
                            lst.get("content", ""), "html.parser"
                        )
                        lists_text += (
                            f"\n\n### {lst.get('text')}\n"
                            f"{list_soup.get_text(separator=chr(10))}"
                        )

                    markdown_text = (
                        f"# Job Title: {data.get('text')}\n"
                        f"**Company/Site:** {site}\n"
                        f"**Location:** {data.get('categories', {}).get('location', 'N/A')}\n"
                        f"**Department:** {data.get('categories', {}).get('department', 'N/A')}\n\n"
                        f"## Overview\n{description}"
                        f"{lists_text}"
                    )
                    return {
                        "text": markdown_text,
                        "strategy": "lever_api",
                        "title": data.get("text", "Job Posting"),
                    }
        except Exception as e:
            print(f"[Warning] Lever API failed: {e}. Falling back to Jina.")
        return None

    def _extract_jina(self, url: str) -> Optional[Dict[str, Any]]:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {**self.headers, "Accept": "text/markdown"}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(jina_url, headers=headers)
                if res.status_code == 200 and res.text.strip():
                    return {
                        "text": res.text,
                        "strategy": "jina_reader",
                        "title": url.split("//")[-1].split("/")[0],
                    }
        except Exception as e:
            print(f"[Warning] Jina Reader failed: {e}. Falling back to local.")
        return None

    def _extract_trafilatura(self, url: str) -> Dict[str, Any]:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                result = trafilatura.extract(
                    downloaded,
                    include_links=False,
                    include_formatting=True,
                    output_format="txt",
                )
                if result:
                    return {
                        "text": result,
                        "strategy": "trafilatura_local",
                        "title": url,
                    }
        except Exception as e:
            print(f"[Error] Trafilatura failed: {e}")

        return {"text": "", "strategy": "failed", "title": url}

    def fetch_url(self, url: str) -> Dict[str, Any]:
        url = url.strip()
        if "greenhouse.io" in url:
            gh_res = self._extract_greenhouse(url)
            if gh_res:
                return gh_res
        elif "lever.co" in url:
            lever_res = self._extract_lever(url)
            if lever_res:
                return lever_res

        jina_res = self._extract_jina(url)
        if jina_res:
            return jina_res

        return self._extract_trafilatura(url)