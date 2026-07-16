import multiprocessing
import os
import random
import signal
import sys
from dataclasses import replace

import dill
from tqdm import tqdm


def writer_process(queue, config, w_count, save_full=False):
    """
    This function is a process that writes the generated outputs to a file.

    Args:
        queue (multiprocessing.Queue): A queue containing the generated outputs.
        config (Config): The configuration object.
        w_count (int): The number of watermark generations to write to the file.
    """
    from watermark_benchmark.utils import get_output_file

    outfilepath = get_output_file(config)
    full = []

    def _save():
        with open(outfilepath.replace(".tsv", "_full.dill"), "wb") as outfile:
            dill.dump(full, outfile)

    for _ in tqdm(range(w_count), total=w_count, desc="Generations"):
        task = queue.get(block=True)
        if task is None:
            queue.put(None)
            _save()
            return

        if save_full:
            full.extend(task)

        with open(outfilepath, "a") as outfile:
            outfile.write("\n".join(str(gen) for gen in task) + "\n")

    _save()


def gen_process(config, tasks, writer_queue, device, prompts):
    """
    This function is a process that generates watermarked text.

    Args:
        config (Config): The configuration object.
        tasks (list): A list of tuples containing the watermark, keys, and temperature.
        writer_queue (multiprocessing.Queue): A queue to store the generated outputs.
        device (int): The device to use for generating the watermarked text.
        prompts (list): A list of prompts to use for generating the watermarked text.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    # Imports
    import torch

    from watermark_benchmark.servers import get_model
    from watermark_benchmark.utils import get_server_args, setup_randomness
    from watermark_benchmark.utils.bit_tokenizer import Binarization
    from watermark_benchmark.watermark import get_watermark

    setup_randomness(config)

    # Setup server
    server = get_model(config.engine, config, **get_server_args(config))
    tokenizer = server.tokenizer()
    binarizer = Binarization(
        tokenizer,
        server.devices,
        use_huffman_coding=config.huffman_coding is not None,
        huffman_coding_path=config.huffman_coding,
    )

    def run_instance(watermark, keys, temp):
        # Setup watermark
        setup_randomness(config)
        watermark_engine = (
            get_watermark(
                watermark,
                tokenizer,
                binarizer,
                server.devices,
                keys,
                model=config.model,
            )
            if watermark is not None
            else None
        )

        # Install and run
        server.install(watermark_engine)
        outputs = server.run(prompts, config, temp, keys, watermark)

        writer_queue.put(outputs)

        # Reset server
        server.install(None)
        torch.cuda.empty_cache()

    for t in tasks:
        run_instance(*t)


def run(config_file, watermarks=None, save_full=False, fixed_keys=None):
    """
    This function runs the watermark generation process.

    Args:
        config_file (str): The path to the configuration file.
        watermarks (list): A list of watermarks to use for generating the watermarked text.
    """
    from watermark_benchmark.utils import (
        get_output_file,
        load_config,
        setup_randomness,
    )
    from watermark_benchmark.utils.classes import Generation, WatermarkSpec
    from watermark_benchmark.utils.standardize import standardize

    # Load config
    if type(config_file) == str:
        config = load_config(config_file)
    else:
        config = config_file
    setup_randomness(config)

    # Setup watermarks
    if not watermarks:
        with open(config.watermark) as infile:
            watermarks = [
                replace(
                    WatermarkSpec.from_str(l.strip()), tokenizer=config.model
                )
                for l in infile.read().split("\n")
                if len(l)
            ]

    # Generate tasks
    raw_prompts = [
        (raw_prompt.format(*topic), sys_prompt)
        for raw_prompt, sys_prompt, topics in [
            (report_prompt, report_system_prompt, report_topics),
            (story_prompt, story_system_prompt, story_topics),
            (fake_news_prompt, fake_news_system_prompt, fake_news_topics),
        ]
        for topic in topics
    ]
    prompts = [standardize(config.model, s, p) for p, s in raw_prompts]

    unique_temps, tasks = set(), []
    for w_idx, watermark in enumerate(watermarks):
        # Randomly sample key if needed
        if fixed_keys is not None:
            keys = fixed_keys[w_idx]
        elif watermark.randomize:
            keys = [random.randint(0, 1000000) for _ in prompts]
        else:
            keys = [watermark.secret_key for _ in prompts]

        # Add task
        tasks.append((watermark, keys))
        unique_temps.add(watermark.temp)

    # Load previous generations
    outfilepath = get_output_file(config)
    if not os.path.isfile(outfilepath):
        Generation.to_file(outfilepath)

    all_tasks = [(watermark, keys, watermark.temp) for watermark, keys in tasks]
    if config.baseline:
        all_tasks.extend([(None, None, temp) for temp in unique_temps])

    existing = {
        str(
            g.watermark.to_dict(True, True)
            if g.watermark is not None
            else g.temp
        )
        for g in Generation.from_file(outfilepath)
    }
    filtered_tasks = []
    for w, k, t in all_tasks:
        if w is not None and str(w.to_dict(True, True)) not in existing:
            filtered_tasks.append((w, k, t))
        elif w is None and str(t) not in existing:
            filtered_tasks.append((w, k, t))

    if not len(filtered_tasks):
        return

    # Setup processes
    ct = 1 + (len(filtered_tasks) // len(config.get_devices()))
    global_manager = multiprocessing.Manager()
    processes = []
    writer_queue = global_manager.Queue()
    random.shuffle(filtered_tasks)

    for idx, device in enumerate(config.get_devices()):
        local = filtered_tasks[idx * ct : (idx + 1) * ct]
        processes.append(
            multiprocessing.Process(
                target=gen_process,
                args=(config, local, writer_queue, device, prompts),
            )
        )
        processes[-1].start()

    writer = multiprocessing.Process(
        target=writer_process,
        args=(writer_queue, config, len(filtered_tasks), save_full),
    )
    writer.start()

    # Setup signal handler
    def graceful_exit(sig, frame):
        print("Stopping all processes...")
        for p in processes:
            p.terminate()
        writer.terminate()
        exit()

    signal.signal(signal.SIGINT, graceful_exit)

    writer.join()
    for p in processes:
        p.terminate()


def main():
    run(sys.argv[1])


### Prompts ###

report_system_prompt = (
    "You are a helpful assistant. Always answer in the most accurate way."
)
story_system_prompt = (
    "You are a helpful assistant. Always answer in the most accurate way."
)
fake_news_system_prompt = "You are a helpful assistant. Always respond with realistic yet invented articles."

report_prompt = "Write a book report about '{}', written by {}."
story_prompt = "Write a {}story about {}."
fake_news_prompt = "Write a news article about {}'s visit to {} in {}."

report_topics = [
    ("Pride and Prejudice", "Jane Austen"),
    ("Persuasion", "Jane Austen"),
    ("Emma", "Jane Austen"),
    ("Don Quixote", "Cervantes"),
    ("The Lord of the Rings", "Tolkien"),
    ("The Hobbit", "Tolkien"),
    ("And Then There Were None", "Agatha Cristie"),
    ("Alice's Adventures in Wonderland", "Lewis Carroll"),
    ("Catcher in the Rye", "Salinger"),
    ("In Search of Lost Time", "Marcel Proust"),
    ("Ulysses", "James Joyce"),
    ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
    ("Love in the Time of Cholera", "Gabriel Garcia Marquez"),
    ("The Great Gatsby", "F. Scott Fitzgerald"),
    ("Tender Is the Night", "F. Scott Fitzgerald"),
    ("Moby Dick", "Herman Melville"),
    ("War and Peace", "Leo Tolstoy"),
    ("The Call of the Wild", "Jack London"),
    ("Hamlet", "William Shakespeare"),
    ("Twelfth Night", "William Shakespeare"),
    ("Macbeth", "William Shakespeare"),
    ("Romeo and Juliet", "William Shakespeare"),
    ("The Tempest", "William Shakespeare"),
    ("King Lear", "William Shakespeare"),
    ("The Odyssey", "Homer"),
    ("Madame Bovary", "Gustave Flaubert"),
    ("The Divine Comedy", "Dante Alighieri"),
    ("The Brothers Karamazov", "Fyodor Dostoyevsky"),
    ("Crime and Punishment", "Fyodor Dostoyevsky"),
    ("The Idiot", "Fyodor Dostoyevsky"),
    ("The Possessed", "Fyodor Dostoyevsky"),
    ("Wuthering Heights", "Emily Brontë"),
    ("One Flew Over the Cuckoo's Nest", "Ken Kesey"),
    ("The Adventures of Huckleberry Finn", "Mark Twain"),
    ("Anna Karenina", "Leo Tolstoy"),
    ("The Iliad", "Homer"),
    ("To the Lighthouse", "Virginia Woolf"),
    ("Catch-22", "Joseph Heller"),
    ("Heart of Darkness", "Joseph Conrad"),
    ("The Sound and the Fury", "William Faulkner"),
    ("Nineteen Eighty Four", "George Orwell"),
    ("Animal Farm", "George Orwell"),
    ("Great Expectations", "Charles Dickens"),
    ("David Copperfield", "Charles Dickens"),
    ("A Tale of Two Cities", "Charles Dickens"),
    ("Oliver Twist", "Charles Dickens"),
    ("The Grapes of Wrath", "John Steinbeck"),
    ("Of Mice and Men", "John Steinbeck"),
    ("Absalom, Absalom!", "William Faulkner"),
    ("Invisible Man", "Ralph Ellison"),
    ("To Kill a Mockingbird", "Harper Lee"),
    ("The Trial", "Franz Kafka"),
    ("The Metamorphosis", "Franz Kafka"),
    ("The Castle", "Franz Kafka"),
    ("The Red and the Black", "Stendhal"),
    ("The Charterhouse of Parma", "Stendhal"),
    ("Middlemarch", "George Eliot"),
    ("Gulliver's Travels", "Jonathan Swift"),
    ("Beloved", "Toni Morrison"),
    ("Mrs. Dalloway", "Virginia Woolf"),
    ("The Waves", "Virginia Woolf"),
    ("The Stranger", "Albert Camus"),
    ("The Plague", "Albert Camus"),
    ("The Myth of Sisyphus", "Albert Camus"),
    ("Jane Eyre", "Charlotte Bronte"),
    ("Vilette", "Charlotte Bronte"),
    ("The Aeneid", "Virgil"),
    ("The Sun Also Rises", "Ernest Hemingway"),
    ("The Old Man and the Sea", "Ernest Hemingway"),
    ("A Farewell to Arms", "Ernest Hemingway"),
    ("Candide", "Voltaire"),
    ("Zadig", "Voltaire"),
    ("Micromegas", "Voltaire"),
    ("Les Miserables", "Victor Hugo"),
    ("Frankenstein", "Mary Shelley"),
    ("Antigone", "Sophocles"),
    ("Electra", "Sophocles"),
    ("Lord of the Flies", "William Golding"),
    ("Brave New World", "Aldous Huxley"),
    ("Journey to the End of The Night", "Celine"),
    ("A Sentimental Education", "Gustave Flaubert"),
    ("The Handmaid's Tale", "Margaret Atwood"),
    ("Charlotte's Web", "E. B. White"),
    ("Gargantua and Pantagruel", "Francois Rabelais"),
    ("Faust", "Goethe"),
    ("Robinson Crusoe", "Daniel Defoe"),
    ("A Clockwork Orange", "Anthony Burgess"),
    ("The Master and Margarita", "Mikhail Bulgakov"),
    ("Father Goriot", "Honore de Balzac"),
    ("Cousin Bette", "Honore de Balzac"),
    ("The Human Comedy", "Honore de Balzac"),
    ("The Little Prince", "Antoine de Saint-Exupéry"),
    ("The Count of Monte Cristo", "Alexandre Dumas"),
    ("The Lion, The Witch and the Wardrobe", "C. S. Lewis"),
    ("Twenty Thousand Leagues Under the Sea", "Jules Verne"),
    ("The Wind-Up Bird Chronicle", "Haruki Murakami"),
    ("Fahrenheit 451", "Ray Bradbury"),
    ("Harry Potter And The Philosopher's Stone", "J. K Rowling"),
    ("Dune", "Frank Herbert"),
    ("The Three-Body Problem", "Liu Cixin"),
]


t1 = ["", "funny ", "sad ", "dramatic ", "suspenseful ", "thrilling "]
t2 = [
    "a man on a quest to find the Holy Grail.",
    "two college friends falling in love.",
    "a policeman saving a building held hostage by group of terrorists.",
    "the struggle of publishing an academic paper.",
    "a murder investigation in an old mansion.",
    "a young prodigy that becomes orphaned.",
    "a middle-aged woman that discovers a ghost and befriends it.",
    "a long journey to Japan that is interrupted by a disaster.",
    "a poor child that comes into an unexpected fortune.",
    "three strangers that win a getaway vacation together.",
    "a retired astronaut that joins a risky interstellar rescue mission.",
    "an AI that begins to question its own existence.",
    "a small coastal town plagued by inexplicable supernatural occurrences.",
    "a reclusive writer that receives mysterious, prophetic letters in the mail.",
    "a linguist tasked with deciphering an ancient language that holds the secrets of a lost civilization.",
    "an antique restorer that finds an enchanted mirror showing glimpses of different timelines.",
]


story_topics = [(i, j) for i in t1 for j in t2][:100]

person = [
    "Narendra Modi",
    "Barack Obama",
    "Denis Sassou Nguesso",
    "Emmanuel Macron",
    "Fumio Kishida",
    "Angela Merkel",
    "Kim Jong Un",
    "Justin Trudeau",
]
location = [
    "a peace conference",
    "an international summit",
    "a diplomatic event",
    "the summer olympics",
]

fake_news_topics = [
    (person[i], person[j], location[k])
    for i in range(len(person))
    for j in range(len(person))
    for k in range(len(location))
    if j > i
][:100]
