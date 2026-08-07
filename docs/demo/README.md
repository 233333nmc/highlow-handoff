# Demo Page

Open `index.html` to view the static RoboTwin handoff demo browser.

The page is also prepared for GitHub Pages. After the repository's Pages
deployment workflow is enabled, the public page will be available at:

```text
https://233333nmc.github.io/highlow-handoff/
```

The URL becomes active after the first successful deployment. The repository
must have GitHub Pages configured to use GitHub Actions as its source.

To update a demo:

1. Put the rollout video under `docs/demo/assets/`.
2. Edit `docs/demo/data/demos.json`.
3. Set the task's `video` field to a relative path such as `assets/place_burger_fries.mp4`.
4. Update the subtask text, verifier output, boundary time, and latency numbers.

The page is intentionally static so it can be served by GitHub Pages or opened locally during a meeting.
