# TODO

## Quick wins
- Plug bunny app into an inference instance
- Add progress % tracker for training or eval / remote compute jobs
- Setup uow pattern to contain all store instead of importing each individually

## AI productivity gains
- Create skill to consolidate learnings from session and update memory of claude.md then compact context.
- Setup linear and try to trigger fully AI run of bug fix / feature.
- Look at subagent harness with multiple agents
    - architect
    - fe
    - be
    - qa

## Data generator
- Workflow to take sample inputs and then use claude to create training data to finetune a small model.

## LLM API
- Setup llm adapter to either 
- Expose inference for each model via API tab on UI
- Provide an apikey for a user to run inference on their model
- Add rate limiting per user 

# Agent harness design
- Design an agent harness system to create an orchestrator agent that can delegate to subagents tracking task progress as it works.
- Add ability to use local or 3rd party agents.

## Fast E2E tests
- Re-enable fast E2E tests on CI/CD now we have working smoke tests

