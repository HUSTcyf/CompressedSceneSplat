# LangZip: CVPR 2027

The Oleafly project compiles `main.tex` together with `supp.tex` so that cross-references resolve in both directions. Build it with:

```bash
oleafly build -C paper/CVPR2027
```

The compiled document contains the eight-page main manuscript, the LLM Usage Statement and references, followed by the supplementary material. The delivered PDFs split these parts at the supplementary title. Re-export the two PDFs after changes; the complete PDF retains cross-document hyperlinks.

The LLM Usage Statement records the known use of OpenAI Codex to assist with writing and improve the manuscript's clarity and readability. Before submission, authors should review it and add any other uses of generative AI in the research and writing. The NeurIPS-specific checklist is excluded from this CVPR manuscript.
