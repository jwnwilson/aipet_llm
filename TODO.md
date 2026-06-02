# TODO

## Fixes
- Fix runpod eval
    - Test run pod eval locally
- Fix log streaming
- Improve worker performance
- improve run details design
- Fix E2E tests to avoid breaking the platform
- Setup uow pattern to contain all store instead of importing each individually

## UI improvements
- Need better logging visibility and progress visualisation on ui.
    - Logs need to stream to UI.
- After triggering a model run re-direct to run in the UI.

## AI productivity gains

- Create skill to consolidate learnings from session and update memory of claude.md then compact context.
- Setup linear and try to trigger fully AI run of bug fix / feature.

## LLM API
- Setup llm adapter to either 
- Expose inference for each model via API tab on UI
- Provide an apikey for a user to run inference on their model
- Add rate limiting per user 

## Fast E2E tests
- Re-enable fast E2E tests on CI/CD now we have working smoke tests

