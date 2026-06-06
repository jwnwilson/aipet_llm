# TODO

## Quick wins
- Add progress % tracker for training or eval / remote compute jobs
- Re-add vastai as a compute option

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
- Create custom agents
- list of tools
- list of mcps
- Create teams for development


## Fast E2E tests
- Re-enable fast E2E tests on CI/CD now we have working smoke tests

