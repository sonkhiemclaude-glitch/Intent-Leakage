"""Train-only fixed-cell candidate fitting and explicit validation selection.

Samples are application requests/replies, not TLS records. This produces
engineering candidates; no deployability or live cost claim is implied.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math
from .causal_cells import PlannedReply, run_logical_session


@dataclass(frozen=True)
class ApplicationSample:
    sample_id: str
    split: str
    request: bytes
    response: bytes
    processing_s: float

    def __post_init__(self):
        if not self.sample_id or self.split not in ('train','validation','test'): raise ValueError('invalid sample identity/split')
        if type(self.request) is not bytes or type(self.response) is not bytes: raise ValueError('application bytes required')
        if not math.isfinite(self.processing_s) or self.processing_s < 0: raise ValueError('invalid processing time')


def score_candidate(sample, *, cell_bytes, period_s, rounds):
    result = run_logical_session([(0., sample.request)], lambda _: PlannedReply(sample.response, sample.processing_s),
                                 cell_bytes=cell_bytes, period_s=period_s, rounds=rounds)
    original = len(sample.request)+len(sample.response)
    return {'sample_id':sample.sample_id,'split':sample.split,'overflow':not result['success'],
            'scheduled_application_bytes':2*rounds*cell_bytes,
            'application_byte_overhead_ratio':2*rounds*cell_bytes/original-1 if original else None,
            'nominal_duration_s':rounds*period_s,'live_cost_measured':False}


def fit_candidates(training, *, cell_sizes=(128,256,512,1024), periods=(.005,.01,.02), margin=1.25, max_rounds=4096):
    if not training or any(s.split!='train' for s in training): raise ValueError('fitting accepts training samples only')
    if len({s.sample_id for s in training})!=len(training): raise ValueError('duplicate training IDs')
    if not math.isfinite(margin) or margin<1: raise ValueError('invalid margin')
    proof=hashlib.sha256(json.dumps([{'id':s.sample_id,'request_sha256':hashlib.sha256(s.request).hexdigest(),
        'response_sha256':hashlib.sha256(s.response).hexdigest(),'processing_s':s.processing_s} for s in training],sort_keys=True).encode()).hexdigest()
    candidates=[]
    for cell in cell_sizes:
        for period in periods:
            # One-request causal slot bound; never reuse unshaped reply timestamps.
            from .framing import CellCodec
            codec=CellCodec(cell_bytes=cell,channel='CT')
            minimum=1
            for sample in training:
                from .framing import CellKind
                request_cells=len(codec.fragment(CellKind.REQUEST,1,sample.request))
                response_cells=len(codec.fragment(CellKind.RESPONSE,1,sample.response))
                request_complete=(request_cells-1)*period
                first_reply=max(0,math.ceil((request_complete+sample.processing_s)/period-.5-1e-9))
                minimum=max(minimum,first_reply+response_cells)
            rounds=math.ceil(minimum*margin)
            if rounds>max_rounds:continue
            candidates.append({'cell_bytes':cell,'period_s':period,'rounds':rounds,
                'termination_s':rounds*period,'training_ids':[s.sample_id for s in training],
                'training_sha256':proof,'margin':margin,'deployable':False,'source':'application_training_samples'})
    return candidates


def select_on_validation(candidates, validation, *, byte_budget, duration_budget_s):
    if not validation or any(s.split!='validation' for s in validation): raise ValueError('selection requires validation, never test')
    if not candidates or byte_budget<=0 or duration_budget_s<=0: raise ValueError('invalid budget/candidates')
    results=[]
    for i,c in enumerate(candidates):
        if set(c['training_ids'])&{s.sample_id for s in validation}:raise ValueError('training/validation identity collision')
        scores=[score_candidate(s,cell_bytes=c['cell_bytes'],period_s=c['period_s'],rounds=c['rounds']) for s in validation]
        feasible=(2*c['rounds']*c['cell_bytes']<=byte_budget and c['termination_s']<=duration_budget_s and not any(v['overflow'] for v in scores))
        results.append({'candidate':i,'feasible':feasible,'scores':scores})
    feasible=[v for v in results if v['feasible']]
    chosen=min(feasible,key=lambda v:(2*candidates[v['candidate']]['rounds']*candidates[v['candidate']]['cell_bytes'],candidates[v['candidate']]['termination_s']))['candidate'] if feasible else None
    return {'selected_candidate':chosen,'validation_ids':[s.sample_id for s in validation], 'results':results,
            'budget':{'application_bytes':byte_budget,'nominal_duration_s':duration_budget_s},'test_data_used':False}
