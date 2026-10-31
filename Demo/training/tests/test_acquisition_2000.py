import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acquire_vietnam import admissible_metadata, safe_image_url


def test_license_and_location_must_both_be_present():
    meta = {'mime': 'image/jpeg', 'width': 1280, 'height': 720,
            'extmetadata': {'LicenseShortName': {'value': 'CC BY-SA 4.0'},
                            'Artist': {'value': 'Photographer'}}}
    assert admissible_metadata(meta, ['Category:Motorcycles in Vietnam'])
    assert not admissible_metadata(meta, ['Category:Motorcycles in India'])
    meta['extmetadata']['ImageDescription']={'value':'A cat on a parked motorcycle in Mumbai, India.'}
    assert not admissible_metadata(meta,['search:Vietnam motorcycle'])
    meta['extmetadata']['LicenseShortName']['value'] = 'All rights reserved'
    assert not admissible_metadata(meta, ['Category:Motorcycles in Vietnam'])


def test_untrusted_download_urls_and_tiny_images_are_rejected():
    assert safe_image_url('https://upload.wikimedia.org/wikipedia/commons/a/a1/Test.jpg')
    assert safe_image_url('https://thumb.wikimedia.org/wikipedia/commons/thumb/a/a1/Test.jpg/1280px-Test.jpg')
    for url in ['http://upload.wikimedia.org/x', 'https://upload.wikimedia.org.evil/x',
                'file:///C:/Windows/win.ini', 'https://127.0.0.1/x']:
        assert not safe_image_url(url)
    assert not admissible_metadata({'mime': 'image/jpeg', 'width': 100, 'height': 80},
                                   ['Category:Motorcycles in Vietnam'])


def test_rate_limit_stops_further_network_requests(monkeypatch,tmp_path):
    import threading
    import pytest
    import acquire_vietnam as module
    calls=[]
    class Response:
        status_code=429
        headers={'Retry-After':'120'}
        def __enter__(self):return self
        def __exit__(self,*args):return False
    def stream(*args,**kwargs):
        calls.append(args)
        return Response()
    monkeypatch.setattr(module,'OUT',tmp_path)
    monkeypatch.setattr(module,'PAUSE',threading.Event())
    monkeypatch.setattr(module.httpx,'stream',stream)
    row={'pageid':1,'url':'https://upload.wikimedia.org/a.jpg'}
    with pytest.raises(module.RateLimitPause) as error:
        module.download(row)
    assert error.value.retry_after==120
    with pytest.raises(module.RateLimitPause):
        module.download({'pageid':2,'url':row['url']})
    assert len(calls)==1
    assert not list(tmp_path.rglob('*.part'))
