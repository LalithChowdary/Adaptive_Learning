"""Frozen checkpoint adapter shared by UI and a future policy consumer."""
import hashlib
import sys
from pathlib import Path
import torch

MODEL_ROOT = Path(__file__).resolve().parents[1] / 'model'
sys.path.insert(0,str(MODEL_ROOT))
from memory.features import EvidenceStream, task_features, DAY
from memory.recurrent import LearnerGRU

CHECKPOINT = MODEL_ROOT / 'runs/v2.1-mps-main/time_decay_gru/seed_20261111/model.pt'

class MemoryService:
    def __init__(self, device='auto'):
        self.device=('mps' if torch.backends.mps.is_available() else 'cpu') if device=='auto' else device
        if self.device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS is not available in this process.')
        payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        if payload['dataset_version']!='2.1.0' or payload['mode']!='time_decay': raise ValueError('Unexpected checkpoint contract')
        self.model=LearnerGRU(mode=payload['mode'],hidden_size=payload['hidden_size']).to(self.device)
        self.model.load_state_dict(payload['weights']); self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.version=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        self.cache={}

    @torch.inference_mode()
    def state(self, events, key=None):
        """Cache by immutable append-only cursor; rebuild after server restart."""
        cached=self.cache.get(key) if key else None
        if cached and cached['count']<=len(events) and (not cached['count'] or cached['last_id']==events[cached['count']-1]['id']):
            stream=cached['stream']; state=cached['state']; anchor=cached['anchor']; start=cached['count']
        else:
            stream=EvidenceStream(); state=self.model.initial_state(); anchor=None; start=0
        for event in events[start:]:
            evidence=stream.push(event['features'])
            if evidence:
                gap=0 if anchor is None else (evidence.time-anchor)/DAY
                state=self.model.observe(state,torch.tensor([evidence.vector()],dtype=torch.float32,device=self.device),torch.tensor([gap],dtype=torch.float32,device=self.device))
                anchor=evidence.time
        result=dict(stream=stream,state=state,anchor=anchor,count=len(events),last_id=events[-1]['id'] if events else None)
        if key: self.cache[key]=result
        return result

    @torch.inference_mode()
    def predict(self, events, task, now, key=None):
        saved=self.state(events,key)
        anchor=saved['anchor']
        gap=0 if anchor is None else max(0,(now-events[0]['at']-anchor)/DAY)
        task_tensor=torch.tensor([task_features(task)],dtype=torch.float32,device=self.device)
        return torch.sigmoid(self.model.predict_logit(saved['state'],task_tensor,torch.tensor([gap],dtype=torch.float32,device=self.device))).item()
