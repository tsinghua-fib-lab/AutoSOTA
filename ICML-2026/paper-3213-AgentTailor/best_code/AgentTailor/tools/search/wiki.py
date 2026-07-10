#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wikipedia
import asyncio

class WikiSearch:
    def __init__(self):
        self.name = "Wikipedia SearchEngine"
        self.description = "Seach for an item in Wikipedia"

    def search(self, query):
        result = wikipedia.search(query[:300], results=1, suggestion=True)
        if len(result[0]) != 0:
            return wikipedia.page(title=result[0]).content
        
        if result[1] is not None:
            result = wikipedia.search(result[1], results=1)
            return wikipedia.page(title=result[0]).content
        
        return None

async def get_wikipedia_summary(title):
    try:
        wikipedia.set_lang("en")
        summ = wikipedia.summary(title)
        return summ
    except wikipedia.exceptions.DisambiguationError as e:
        return await get_wikipedia_summary(e.options[0])
    except wikipedia.exceptions.PageError:
        return ""

async def search_wiki(query):
    wikipedia.set_lang("en")
    result = wikipedia.search(query, results=2, suggestion=True)
    ret = ""
    tasks = []
    
    if len(result[0]) != 0:
        for res in result[0]:
            tasks.append(get_wikipedia_summary(res))
        summaries = await asyncio.gather(*tasks)
        for res, summa in zip(result[0], summaries):
            if len(summa):
                ret += f"The summary of {res} in Wikipedia is: {summa}\n"
    if result[1] is not None:
        summa = await get_wikipedia_summary(result[1])
        if len(summa):
            ret += f"The summary of {result[1]} in Wikipedia is: {summa}\n"
    return ret


async def search_wiki_main(queries):
    tasks = [search_wiki(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results

if __name__ == "__main__":
    queries = ["Python", "Asyncio", "Wikipedia"]
    asyncio.run(search_wiki_main(queries))