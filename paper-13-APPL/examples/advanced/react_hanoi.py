import textwrap

import numpy as np

import appl
from appl import (
    AIMessage,
    AIRole,
    Generation,
    as_tool,
    as_tool_choice,
    gen,
    ppl,
    records,
)


# TowerOfHanoi mostly written by GPT4
class TowerOfHanoi:
    """
    This class provides a simple implementation of the Tower of Hanoi game, allowing the
    configuration of the number of disks and tracking the state of the game through moves.

    The Tower of Hanoi is a mathematical puzzle where the objective is to move a stack
    of disks from one peg to another, with the help of one auxiliary peg, adhering to the
    following rules:
    1. Only one disk can be moved at a time.
    2. Each move consists of taking the upper disk from one of the stacks and placing it
        on top of another stack or on an empty peg.
    3. No disk may be placed on top of a smaller disk.

    Attributes:
        num_disks (int): The number of disks in the game.
        pegs (list[list[int]]): The state of the pegs and disks.
    """

    def __init__(self, num_disks: int):
        """
        Initializes the Tower of Hanoi game with the given number of disks.

        Args:
            num_disks (int): The number of disks to use in the game.
        """
        self.num_disks = num_disks
        self.reset()

    def reset(self) -> None:
        # Initialize the pegs; start with all disks on peg 1
        self.pegs: list[list[int]] = [list(range(self.num_disks, 0, -1)), [], []]

    def move_disk(self, from_peg: int, to_peg: int) -> bool:
        """
        Moves the top disk from one peg to another if the move is valid.

        Args:
            from_peg (int): The peg number (0, 1, 2) to move the disk from.
            to_peg (int): The peg number (0, 1, 2) to move the disk to.

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        # Check if move is valid
        if (
            from_peg == to_peg
            or from_peg < 0
            or from_peg > 2
            or to_peg < 0
            or to_peg > 2
            or not self.pegs[from_peg]
            or (self.pegs[to_peg] and self.pegs[from_peg][-1] > self.pegs[to_peg][-1])
        ):
            return False

        # Move the disk
        self.pegs[to_peg].append(self.pegs[from_peg].pop())
        return True

    def is_solved(self) -> bool:
        """
        Checks if the game is solved, i.e., all disks are moved to the third peg.

        Returns:
            bool: True if the game is solved, False otherwise.
        """
        return len(self.pegs[2]) == self.num_disks

    def render(self) -> str:
        """
        Render the current state of the pegs and disks.
        """
        s = ""
        for i, peg in enumerate(self.pegs):
            s += f"Peg {i}: " + " ".join(str(disk) for disk in peg) + "\n"
        return s


def move_disk(env: TowerOfHanoi, from_peg: int, to_peg: int) -> str:
    # wrap the move_disk method to be used as a tool
    """Moves the top disk from one peg to another if the move is valid.

    Args:
        from_peg (int): The peg number (0, 1, 2) to move the disk from.
        to_peg (int): The peg number (0, 1, 2) to move the disk to.

    Returns:
        str:
            The current state of the pegs and disks if the move was
            successful, "Invalid move" otherwise.
    """
    if env.move_disk(from_peg, to_peg):
        return env.render()
    return "Invalid move"


@ppl
def react(env: TowerOfHanoi, max_steps: int = 10):
    tools = [as_tool(move_disk, env=env)]
    desc = (TowerOfHanoi.__doc__ or "\n\n").split("\n\n")[1]
    (desc := textwrap.dedent(desc))  # add desc to prompt

    "Before taking actions in each step, you should briefly write thoughts for the next step."
    "The initial state of the pegs and disks is:\n"
    env.render()  # render, add to prompt
    steps = 0
    while steps < max_steps:
        with AIRole():
            f"Thoughts: {gen(tools=tools, tool_choice='none')}"
        (actions := gen(tools=tools, tool_choice=as_tool_choice(move_disk)))
        if actions.is_tool_call:
            (results := actions.run_tool_calls())  # results is a list of ToolMessage
            for i in range(len(results)):
                steps += 1
                print(f"step {steps}: {results[i].get_content()}")
        else:
            assert False, "Tool call expected."
        if env.is_solved():
            print(f"Solved in {steps} steps!")
            break


if __name__ == "__main__":
    env = TowerOfHanoi(2)
    env.reset()
    react(env)
