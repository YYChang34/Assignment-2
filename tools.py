import os
import requests
import json


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API and return summarized results.
    This tool will be used by the ReAct agent.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return json.dumps({
            "ok": False,
            "error": "TAVILY_API_KEY not set in environment",
            "query": query,
            "results": []
        })

    if not query or len(query.strip()) == 0:
        return json.dumps({
            "ok": False,
            "error": "Empty query",
            "results": []
        })

    url = "https://api.tavily.com/search"

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results
    }

    try:
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code != 200:
            return json.dumps({
                "ok": False,
                "error": f"HTTP {response.status_code}",
                "query": query,
                "results": []
            })

        data = response.json()

        results = []

        for r in data.get("results", []):
            results.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content")
            })

        return json.dumps({
            "ok": True,
            "query": query,
            "answer": data.get("answer"),
            "results": results
        }, indent=2)

    except requests.exceptions.RequestException as e:
        return json.dumps({
            "ok": False,
            "error": str(e),
            "query": query,
            "results": []
        })

    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"Unexpected error: {str(e)}",
            "query": query,
            "results": []
        })