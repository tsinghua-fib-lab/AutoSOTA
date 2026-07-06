class BaseGame:
    def num_players(self):
        raise NotImplementedError()

    def num_actions(self, player_id):
        raise NotImplementedError()

    def full_feedback(self, strategies):
        raise NotImplementedError()

    def nash_conv(self, strategies):
        raise NotImplementedError()
