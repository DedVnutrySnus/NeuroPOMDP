# Model

NeuroPOMDP uses a discrete partially observable decision-making task with categorical probability tables.

## Hidden States

The benchmark uses eight hidden states:

```text
0 left secret
1 right secret
2 left revealed
3 right revealed
4 left success
5 left failure
6 right success
7 right failure
```

The initial prior places probability mass on the two secret states. The `prior_left_prob` configuration controls the prior probability of the left secret state.

## Observations

The model uses five observations:

```text
0 ambiguous
1 left cue
2 right cue
3 success
4 failure
```

The observation model is controlled by `observation_noise` and `information_quality`. Higher observation noise makes cues and terminal observations less reliable.

## Actions

The action set contains:

```text
0 inspect
1 exploit left
2 exploit right
```

The inspect action can reveal information before a later exploit action. Exploit actions transition to success or failure states depending on the hidden state and transition stochasticity.

## Generative Model Tables

`A` maps hidden states to observations.

`B` maps previous states and actions to next states.

`C` encodes preferences over observations as a categorical distribution.

`D` encodes the initial prior over hidden states.

The model builder validates dimensions, non-negativity, and categorical normalization before returning a model instance.

## Expected Free Energy

Action selection combines pragmatic cost with epistemic value. The no-epistemic baseline disables the epistemic component, which makes it useful for measuring when information seeking provides an advantage.
