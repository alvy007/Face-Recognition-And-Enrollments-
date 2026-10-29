import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from app import annotate

def test_render_with_known_face():
    frame=np.zeros((240,320,3),dtype=np.uint8)
    face=np.array([40,40,70,80, 50,50,60,50,55,60,52,72,64,72, .9],dtype=np.float32)
    canvas=annotate(frame,[face],[np.array([1.,0.])],{'Alice':np.array([1.,0.])},.363)
    assert canvas.shape==frame.shape
    assert canvas.sum()>0
    assert frame.sum()==0  # original immutable

def test_render_no_enrollment():
    frame=np.zeros((240,320,3),dtype=np.uint8)
    face=np.array([40,40,70,80]+[0]*11,dtype=np.float32)
    assert annotate(frame,[face],[np.array([1.,0.])],{},.363).sum()>0
