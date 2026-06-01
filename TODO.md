# TODO

## Validate Inference
- Add test to load a small model and validate inference is working, currently tests only check for a response from inference which will defaut to IDLE if it fails.

## UI improvements
- Setup BE / FE consistent pagination for all listing compnents
    - Good potential fully ai job 
- Need better logging visibility and progress visualisation on ui.
    - Logs need to stream to UI.
- Group inference under a model and version the different inferences to see the latest inferences for each model
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

