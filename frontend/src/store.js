import {api} from './api.js';
export const store={user:null,probes:[],sweeps:[],layouts:[],certificates:[],runs:[],mounts:[],runMounts:[],batches:[],deliveries:[],curves:{},archives:[],layoutId:null,probeId:1};
export async function refresh(){
 [store.probes,store.sweeps,store.layouts,store.certificates,store.runs,store.mounts,store.runMounts,store.batches,store.deliveries,store.archives]=await Promise.all([api('/probes'),api('/sweeps'),api('/layouts'),api('/calibration-certificates'),api('/tunnel-runs'),api('/model-mounts'),api('/run-mounts'),api('/acquisition-batches'),api('/tunnel-deliveries'),api('/lift-model-archives')]);
 store.curves=Object.fromEntries(await Promise.all(store.batches.map(async b=>[b.id,await api('/acquisition-batches/'+b.id+'/lift-curves')])));
 if(!store.layoutId&&store.layouts.length)store.layoutId=store.layouts[0].id;return store}
export const layout=s=>s.layouts.find(x=>x.id===Number(s.layoutId));
