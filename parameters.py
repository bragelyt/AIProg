Parameters = {
    "useTableCritic": True,
    "useDiamondBoard": False,
    "showFinalSolution": True,
    "actorLearningRate": 0.75,
    "actorDiscFactor": 0.9,
    "actorTraceDecay": 0.9,
    "criticLearningRate": 0.75,
    "criticDiscFactor": 0.9,
    "criticTraceDecay": 0.8,
    "epsilon": 0.8,
    "epsilonDecay": 0.994,
    "numberOfEpisodes": 700,
    "frameDelay": 500,
    "architecture": [15, 20, 5, 1],
}
# examples:

# eDecay = 0.99
# nEpisodes <= 400 ish

# eDecay = 0.994
# nEpisodes >= 900

# 0.997, 1500

DiamondBoard = {
    "emptyStartPins": [6],
    "size": 4
}

TriangleBoard = {
    "emptyStartPins": [0],
    "size": 5
}
