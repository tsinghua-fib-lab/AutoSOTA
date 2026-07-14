import pytest 
from tempfile import TemporaryFile, NamedTemporaryFile
from RCAEval.logparser import EventTemplate


@pytest.mark.parametrize("pattern, keyset, logkeyset", [
    (
        {
            "key-a": {
                "key-b": {
                    "key-c": ""
                }
            },
            "key-d": ""
        }, 
        {
            "key-a",
            "key-a.key-b",
            "key-a.key-b.key-c",
            "key-d"
        },
        set()
     ),
    (
        {
            "key-a": {
                "key-b": {
                    "key-c": {
                        "key-d": "",
                        "key-e": "UserId=<:ID:>"
                    }
                }
            },
            "key-e": "This is a log"
        },
        {
            "key-a",
            "key-a.key-b",
            "key-a.key-b.key-c",
            "key-a.key-b.key-c.key-d",
            "key-a.key-b.key-c.key-e",
            "key-e"
        },
        {
            "key-e",
            "key-a.key-b.key-c.key-e"
        }
    ),
])
def test_find_key(pattern, keyset, logkeyset):
    e = EventTemplate(pattern)
    assert e.keyset == keyset
    assert e.logkeyset == logkeyset


@pytest.mark.parametrize("pattern, no_matches, matches", [
    (
        {
            "key-a": "",
            "key-b": ""
        },
        [
            {
                "key-a": "value-a",
                "key-c": "value-c"
            }
        ],
        [
            {
                "key-a": "value-a",
                "key-b": "value-b"
            }
        ]
    ),
])
def test_load_and_match_template(pattern, no_matches, matches):
    template = EventTemplate(pattern)
    
    # Test strings that should NOT match
    for no_match in no_matches:
        assert not template.is_match(no_match)
    
    # Test strings that should match
    for match in matches:
        assert template.is_match(match)


#def test_load_multiple_templates():
#    temfile = TemporaryFile(mode='w+')
#    temfile.write(
#        "# This is a comment\n"
#        "received ad request (context_words=[<*>])\n"
#        "SEVERE: Exception while executing runnable <*>"
#    )
#    temfile.seek(0)
#    templates = LogTemplate.load_templates_from_txt(template_file=temfile.name)
#    assert len(templates) == 2
#    assert isinstance(templates[0], LogTemplate)
#    assert isinstance(templates[1], LogTemplate)
#    assert templates[0].template == "received ad request (context_words=[<*>])"
#    assert templates[1].template == "SEVERE: Exception while executing runnable <*>"
#
#
#def test_file_matching():
#    template_file = NamedTemporaryFile(mode='w+', suffix=".txt")
#    template_file.write(
#        "# This is a comment\n"
#        "received ad request (context_words=[<*>])\n"
#        "SEVERE: Exception while executing runnable <*>"
#    )
#    template_file.seek(0)
#
#    log_file = TemporaryFile(mode='w+', suffix=".log")
#    log_file.write(
#        "received ad request (context_words=[clothing])\n"
#        "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e\n"
#        "Match no thing\n"
#        "received ad request (context_words=[])\n"
#    )
#    log_file.seek(0)
#
#    df = LogTemplate.parse_logs(
#        template_file=template_file.name,
#        log_file=log_file.name,
#    )
#
#    # df should have two columns ('log', and 'event type'), each rows = each log with the correspndin template
#    assert df.shape[0] == 4
#    assert df.shape[1] == 2
#    assert df.columns.tolist() == ['log', 'event type']
#    assert df.iloc[0]['log'] == "received ad request (context_words=[clothing])"
#    assert df.iloc[0]['event type'] == "received ad request (context_words=[<*>])"
#
#
#
#def test_detect_multiple_template():
#    """one log may match multiple templates, we need to detect this case"""
#    template_file = NamedTemporaryFile(mode='w+', suffix=".txt")
#    template_file.write(
#        "template 1 <*>\n"
#        "<*> template 2\n"
#    )
#    template_file.seek(0)
#
#    log_file = TemporaryFile(mode='w+')
#    log_file.write(
#        "received ad request (context_words=[clothing])\n"
#        "template 1 template 2\n"
#    )
#    log_file.seek(0)
#
#
#    output = LogTemplate.is_duplicate(
#        template_file=template_file.name,
#        log_file=log_file.name,
#    )
#
#    assert output == True
#
#
#def test_completeness():
#    """ensure all logs are matched"""
#    template_file = NamedTemporaryFile(mode='w+', suffix=".txt")
#    template_file.write(
#        "# This is a comment\n"
#        "received ad request (context_words=[<*>])\n"
#        "SEVERE: Exception while executing runnable <*>"
#    )
#    template_file.seek(0)
#
#    log_file1 = NamedTemporaryFile(mode='w+', suffix=".log")
#    log_file1.write(
#        "received ad request (context_words=[clothing])\n"
#        "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e\n"
#        "Match no thing\n"
#        "received ad request (context_words=[])\n"
#    )
#    log_file1.seek(0)
#
#    output = LogTemplate.is_complete(
#        template_file=template_file.name,
#        log_file=log_file1.name,
#    )
#    assert output == False
#
#    log_file2 = NamedTemporaryFile(mode='w+')
#    log_file2.write(
#        "received ad request (context_words=[clothing])\n"
#        "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e\n"
#        "received ad request (context_words=[])\n"
#    )
#    log_file2.seek(0)
#
#    output = LogTemplate.is_complete(
#        template_file=template_file.name,
#        log_file=log_file2.name,
#    )
#    assert output == True
#
#def test_from_toml():
#    templates = LogTemplate.load_templates_from_toml("tests/data/carts.toml")
#
#    log = """2024-01-18 16:31:07.450  WARN [carts,,,] 1 --- [tion/x-thrift})] z.r.AsyncReporter$BoundedAsyncReporter   : Dropped 2 spans due to UnknownHostException(zipkin)"""
#    valid = False
#    for template in templates:
#        if template.id == "E2" and template.is_match(log):
#            valid = True
#            break
#    assert valid == True
