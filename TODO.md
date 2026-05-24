#TODO

- Fix kaggle and verify vastai runpod training works

## Better llm-api architecture
- Setup functionality to set a model to "active"
- Spin up a container for each "active" llm models requesting the right memory for the model.
- Setup scaling for each model independanlty
    - Scale to 0 if model is not used for 1 hour
- Track status on active llm models to show on the ui.
- Handle requests to loading models and return a result or good http status with "not_ready_yet" 

## LLM API
- Expose inference for each model via API tab on UI
- Provide an apikey for a user to run inference on their model
- Add rate limiting per user 

## Fast E2E tests
- Re-enable E2E tests on CI/CD to run once a day or something

