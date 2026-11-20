"""Conservative content candidates for human review, never automatic clearance."""
from PIL import Image
import numpy as np
from perception_cache import file_sha256
from video_io import inspect_video,iter_frames


def fingerprint_video(path,max_samples=8):
    if max_samples<2:raise ValueError('At least two samples required.')
    info=inspect_video(path);hashes=[];times=[]
    frames=iter_frames(path,1.0)
    try:
        for timestamp,rgb in frames:
            pixels=np.asarray(Image.fromarray(rgb).convert('L').resize((17,16)),dtype=np.float32)
            bits=(pixels[:,1:]>pixels[:,:-1]).reshape(-1)
            hashes.append(f'{int.from_bytes(np.packbits(bits).tobytes(),"big"):064x}')
            times.append(timestamp)
            if len(hashes)>=max_samples:break
    finally:frames.close()
    return {'format':'activa-video-fingerprint-v1','video_id':info.video_id,'sha256':file_sha256(info.path),'duration':info.duration,'width':info.width,'height':info.height,'rotation':info.rotation,'display_width':info.height if info.rotation%180 else info.width,'display_height':info.width if info.rotation%180 else info.height,'times':times,'dhash256':hashes,'scope':'first sampled seconds only; candidate matching cannot clear reencoded/trimmed videos'}


def possible_content_match(first,second,max_hamming=12):
    """Returns a review candidate, not identity proof (flat footage may collide)."""
    for item in (first,second):
        if item.get('format')!='activa-video-fingerprint-v1' or any(not np.isfinite(item.get(key,0)) or item.get(key,0)<=0 for key in ['duration','width','height']):
            raise ValueError('Invalid fingerprint metadata.')
        if len(item.get('times',[]))!=len(item.get('dhash256',[])):
            raise ValueError('Fingerprint times/hash mismatch.')
        if any(len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in item['dhash256']):
            raise ValueError('Invalid dHash256.')
    if abs(first['duration']-second['duration'])>max(1.,first['duration']*.01):return False
    aspect=lambda item:item.get('display_width',item['width'])/item.get('display_height',item['height'])
    if abs(aspect(first)-aspect(second))>.03:return False
    count=min(len(first['dhash256']),len(second['dhash256']))
    if count<2:return False
    if not np.allclose(first['times'][:count],second['times'][:count],atol=.2):return False
    distances=[(int(a,16)^int(b,16)).bit_count() for a,b in zip(first['dhash256'][:count],second['dhash256'][:count])]
    return bool(sum(distance<=max_hamming for distance in distances)>=np.ceil(.75*count))
