import torch

from cpc_llm.data.synthetic_dataset_lib import format_fewshot, ranked_fft


class TestRankedFarthestFirstTraversal:
    def test_euclidean(self):
        library = torch.Tensor(
            [
                [1, 2, 3],  # lowest score, will be selected first
                [1, 2, 4],
                [3, 2, 3],  # furthest from the first row, will be selected 2nd
            ]
        )
        scores = torch.Tensor([-1.0, -0.5, 0.01])
        selected_idx = ranked_fft(library, scores, n=2)
        assert selected_idx.tolist() == [0, 2]

    def test_manhattan(self):
        library = torch.Tensor(
            [
                [3, 4, 3],  # furthest by l1 distance, not by l2 distance
                [1, 2, 3],  # lowest score, will be selected first
                [3.75, 2, 2],
            ]
        )
        scores = torch.Tensor([-0.5, -1.0, 0.01])

        def distance_fn(x, y):
            return torch.norm(x - y, p=1).item()

        selected_idx = ranked_fft(library, scores, n=2, distance_fn=distance_fn)
        assert selected_idx.tolist() == [1, 0]

    def test_descending(self):
        library = torch.Tensor(
            [
                [3, 4, 3],
                [1, 2, 3],  # highest score, will be selected first
                [3.75, 2, 2],  # furthest by l2 distance
            ]
        )
        scores = torch.Tensor([-0.1, 1.0, 0.01])
        selected_idx = ranked_fft(library, scores, n=2, descending=True)
        assert selected_idx.tolist() == [1, 2]


class TestFormatFewshot:
    def test_disallow_same_score(self):
        X = torch.FloatTensor(
            [
                [1.0, 4.0, 2.0, 3.0],
                [2.0, 1.0, 3.0, 5.0],
                [3.0, 3.0, 2.0, 4.0],
            ]
        )
        scores = torch.FloatTensor([-0.5, 0.25, -0.5])
        ks = [1]  # only do one-shot
        outputs = format_fewshot(X, scores, ks, allow_same_score=False, seed=0)
        assert X.shape[0] == len(outputs)
        for o in outputs:
            assert o["input"].count("Score: ") == ks[0] + 1
            assert o["input"].count("Particle: ") == ks[0] + 1
        assert (
            outputs[0]["input"]
            == "Score: 0.250\nParticle: [2, 1, 3, 5]\n\nScore: -0.500\nParticle: "
        )
        assert outputs[0]["target"] == "[1, 4, 2, 3]"
        assert outputs[1]["target"] == "[2, 1, 3, 5]"
        assert (
            outputs[2]["input"]
            == "Score: 0.250\nParticle: [2, 1, 3, 5]\n\nScore: -0.500\nParticle: "
        )
        assert outputs[2]["target"] == "[3, 3, 2, 4]"
