class BaseClient:
    def __init__(self, args, client_id, data, data_dir, message_pool, device, personalized=False):
        from fgl.utils.basic_utils import load_task
        self.args = args
        self.client_id = client_id
        self.message_pool = message_pool
        self.device = device
        self.task = load_task(args, client_id, data, data_dir, device)
        self.personalized = personalized

    def execute(self):
        raise NotImplementedError

    def send_message(self):
        raise NotImplementedError


class BaseServer:
    def __init__(self, args, global_data, data_dir, message_pool, device, personalized=False):
        from fgl.utils.basic_utils import load_task

        self.args = args
        self.message_pool = message_pool
        self.device = device
        self.task = load_task(args, None, global_data, data_dir, device)
        self.personalized = personalized

    def execute(self):
        raise NotImplementedError

    def send_message(self):
        raise NotImplementedError
