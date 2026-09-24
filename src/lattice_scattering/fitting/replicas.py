"""Sequential sample-preserving fits with explicit failure accounting."""
import copy
import hashlib
import json
import numpy as np
from lattice_scattering.data.replicas import ReplicaTable


def fit_replicas(table,fit_sample,*,parameter_ids,max_samples=None,resume=None,run_id=None,on_checkpoint=None):
    """fit_sample(sample_id, observable_mapping) returns a parameter vector.

    The callback constructs the sample's spectrum, masses and anisotropy
    from the same mapping, chooses its root intervals, and performs the fit.
    Fixed versus resampled fit weights must be decided by that callback.
    No central estimate is inferred from the replica mean.

    Stateful calls require a caller-supplied run_id identifying the callback,
    model, bounds, weights and root policy. Resume requires exactly the same
    input table and parameter order. Failed records remain failed; only pending
    samples are executed. A checkpoint callback receives a detached snapshot
    after each attempt; its exceptions propagate rather than becoming fit errors.
    """
    ids=tuple(parameter_ids)
    if not ids or len(set(ids))!=len(ids) or any(not isinstance(x,str) or not x for x in ids):
        raise ValueError('unique nonempty parameter identities required')
    if max_samples is not None and (isinstance(max_samples,bool) or not isinstance(max_samples,(int,np.integer)) or max_samples<1):
        raise ValueError('positive integer sample budget required')
    if (max_samples is not None or resume is not None or on_checkpoint is not None) and (not isinstance(run_id,str) or not run_id.strip()):
        raise ValueError('explicit run_id required for checkpointed execution')
    if run_id is not None and (not isinstance(run_id,str) or not run_id.strip()):raise ValueError('nonempty run_id required')
    if on_checkpoint is not None and not callable(on_checkpoint):raise ValueError('checkpoint callback must be callable')
    payload={'values':table.values.tolist(),'sample_ids':table.sample_ids,'observable_ids':table.observable_ids,
             'kind':table.kind,'ensemble':table.ensemble,'parameter_ids':ids}
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    records=[{'sample_id':sample,'status':'pending'} for sample in table.sample_ids]
    if resume is not None:
        if (not isinstance(resume,dict) or resume.get('input_fingerprint')!=fingerprint or resume.get('run_id')!=run_id
                or not isinstance(resume.get('parameter_ids'),(list,tuple)) or tuple(resume['parameter_ids'])!=ids
                or resume.get('kind')!=table.kind or resume.get('ensemble')!=table.ensemble):
            raise ValueError('checkpoint input or run identity mismatch')
        saved=resume.get('records')
        if not isinstance(saved,list) or len(saved)!=len(records):raise ValueError('checkpoint record count mismatch')
        for i,(record,sample) in enumerate(zip(saved,table.sample_ids)):
            if not isinstance(record,dict) or record.get('sample_id')!=sample:raise ValueError('checkpoint sample order mismatch')
            status=record.get('status')
            if status=='success':
                values=np.asarray(record.get('parameters'))
                if values.shape!=(len(ids),) or np.iscomplexobj(values) or not np.issubdtype(values.dtype,np.number) or not np.all(np.isfinite(values)):
                    raise ValueError('invalid checkpoint parameters')
                records[i]={'sample_id':sample,'status':status,'parameters':values.astype(float).tolist()}
            elif status=='failed':
                if any(not isinstance(record.get(key),str) for key in ('error_type','error')):raise ValueError('invalid checkpoint failure')
                records[i]={key:record[key] for key in ('sample_id','status','error_type','error')}
            elif status!='pending':raise ValueError('invalid checkpoint status')
    attempted=0
    def snapshot():
        complete=all(r['status']=='success' for r in records)
        result={'kind':table.kind,'ensemble':table.ensemble,'parameter_ids':ids,
                'records':copy.deepcopy(records),'complete':complete,'covariance':None,'central_estimate':None,
                'input_fingerprint':fingerprint,'run_id':run_id,'attempted_this_call':attempted,
                'finished':all(r['status']!='pending' for r in records),
                'pending_count':sum(r['status']=='pending' for r in records)}
        if complete:
            output=ReplicaTable([r['parameters'] for r in records],table.sample_ids,ids,table.kind,table.ensemble)
            result['covariance']=output.covariance().tolist()
            result['replica_mean']=output.values.mean(axis=0).tolist()
        return result
    for index,(sample,row) in enumerate(zip(table.sample_ids,table.values)):
        if records[index]['status']!='pending':continue
        if max_samples is not None and attempted>=max_samples:break
        try:
            fitted=np.asarray(fit_sample(sample,dict(zip(table.observable_ids,row))))
            if fitted.shape!=(len(ids),) or np.iscomplexobj(fitted) or not np.all(np.isfinite(fitted)):
                raise ValueError('fit returned invalid real parameter vector')
            records[index]={'sample_id':sample,'status':'success','parameters':fitted.astype(float).tolist()}
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            records[index]={'sample_id':sample,'status':'failed','error_type':type(error).__name__,'error':str(error)}
        attempted+=1
        if on_checkpoint is not None:on_checkpoint(snapshot())
    return snapshot()
