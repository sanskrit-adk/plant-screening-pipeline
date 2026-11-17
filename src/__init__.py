"""Tomato-leaf field screening pipeline.

restoration -> enhancement -> compression -> segmentation/morphology -> CNN
-> cost-sensitive decision layer (CLEAR / FLAG / RE-IMAGE).

Run everything from the project root:  python -m src.experiments --exp <name>
"""
