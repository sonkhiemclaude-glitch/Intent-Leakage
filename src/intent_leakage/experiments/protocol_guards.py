"""Mandatory split, transfer and observation-view checks for real attackers."""
from __future__ import annotations
from collections.abc import Mapping
from ..features.extraction import extract_channel_features, extract_fused_features
from ..splits.grouped import validate_grouped_split, split_assignment_digest
from ..splits.transfer import FrozenTransferProtocol


def validate_real_attacker_protocol(frame, *, feature_columns_by_view: Mapping, split_column: str,
                                    core_topology: str | None, core_model: str | None) -> dict:
    if not core_topology or not core_model:
        raise ValueError('Real fitting requires the frozen core topology and model ID')
    fields=('intent_instance_id','phrase_family_id','topology_id','model_id',split_column)
    missing=set(fields)-set(frame.columns)
    if missing:raise ValueError('Missing real protocol metadata: '+str(sorted(missing)))
    if frame[list(fields)].isna().any().any() or any(frame[f].astype(str).str.strip().eq('').any() for f in fields):
        raise ValueError('Real protocol metadata cannot be empty')
    if not frame[split_column].isin(('train','validation','test')).all():
        raise ValueError('Only frozen train/validation/test splits are accepted')
    validate_grouped_split(frame,split_field=split_column)
    protocol=FrozenTransferProtocol(core_topology=core_topology,core_model=core_model)
    for row in frame[['topology_id','model_id',split_column]].drop_duplicates().to_dict('records'):
        env={'topology':row['topology_id'],'model':row['model_id'],'split':row[split_column]}
        if env['topology']==core_topology and env['model']==core_model:
            if env['split']!='test':protocol.assert_fit_allowed(**env)
        else:protocol.validate_transfer(**env)
    schema={view:set(extract_channel_features([],prefix=view.lower()+'__')) for view in ('CL','CT','CA')}
    schema['fused']=set(extract_fused_features([]))
    if not feature_columns_by_view:raise ValueError('No observation views')
    for view,columns in feature_columns_by_view.items():
        if view not in schema or not columns or set(columns)-schema[view]:
            raise ValueError('Feature columns do not belong to the frozen observation view: '+str(view))
    return {'core_topology':core_topology,'core_model':core_model,'feature_schema':'encrypted_metadata_v1',
            'split_assignment_sha256':split_assignment_digest(frame,split_field=split_column)}
