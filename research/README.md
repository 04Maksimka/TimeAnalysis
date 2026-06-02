# Research blog

The `research/` directory is an independent Quarto website. Its articles are
Jupyter notebooks stored in `research/posts/`.

## Publish a new research notebook

1. Copy `posts/example-research.ipynb` to `posts/<short-name>.ipynb`.
2. Update the YAML metadata in the first Raw cell:

   ```yaml
   ---
   title: "Readable research title"
   description: "One sentence shown on the card."
   author: "Your name"
   date: "2026-06-02"
   categories: [eda, time-series]
   ---
   ```

3. Write the research and run its cells locally.
4. Commit the notebook with its saved outputs and push it to `main`.

The `Publish research blog` GitHub Actions workflow renders the saved outputs
without executing notebook code and deploys the site to:

<https://04Maksimka.github.io/TimeAnalysis/>

## Preview locally

Install [Quarto](https://quarto.org/docs/get-started/) once, then run:

```bash
quarto preview research
```

## Enable GitHub Pages once

Open the repository settings on GitHub, go to **Pages**, and select
**GitHub Actions** as the source under **Build and deployment**.
