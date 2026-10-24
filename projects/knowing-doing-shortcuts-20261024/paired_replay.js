// R004 paired-replay auditor (Node.js 18+). Does not run a model.
// Usage: node paired_replay.js trials.json
// JSON array contains: task,seed,prefix,price,mode,action,hidden
// mode: skip,call,free. Hidden outcome: 0/1.
// Exact predecision replay must be verified externally, not inferred from hashes.
const fs=require("fs");
function audit(rows){
 const groups=new Map();
 for(const x of rows){
   for(const k of ["task","seed","prefix","price","mode","action","hidden"])
     if(x[k]===undefined)throw Error("Missing "+k);
   if(!["skip","call","free"].includes(x.mode))throw Error("Unknown mode");
   if(![0,1].includes(x.action)||![0,1].includes(x.hidden)||x.price<0)
     throw Error("Invalid values");
   if(x.mode==="skip"&&x.action!==0||x.mode==="call"&&x.action!==1)
     throw Error("Forced choice inconsistent");
   const key=JSON.stringify([x.task,x.seed,x.prefix,x.price]);
   if(!groups.has(key))groups.set(key,{});
   const g=groups.get(key);
   if(g[x.mode])throw Error("Duplicate arm "+key);
   g[x.mode]=x;
 }
 const summary={paired_histories:0,no_tool_failure_without_benefit:0,
   harmful_calls:0,helpful_calls:0,dominated_skips:0,mean_realized_regret:0};
 let regret=0;
 for(const [key,g] of groups){
   if(!g.skip||!g.call||!g.free)throw Error("Incomplete replay "+key);
   const delta=g.call.hidden-g.skip.hidden;
   summary.paired_histories++;
   if(g.skip.hidden===0 && delta<=0)summary.no_tool_failure_without_benefit++;
   if(delta<0)summary.harmful_calls++;
   if(delta>0)summary.helpful_calls++;
   if(!g.free.action){
     const loss=Math.max(0,delta-g.free.price);
     if(loss>0)summary.dominated_skips++;
     regret+=loss;
   }
 }
 summary.mean_realized_regret=regret/summary.paired_histories;
 summary.important_caveat="These are single realized outcomes; repeated randomized continuations are necessary to estimate expected effects.";
 return summary;
}
if(require.main===module){
 const rows=JSON.parse(fs.readFileSync(process.argv[2],"utf8"));
 console.log(JSON.stringify(audit(rows),null,2));
}
module.exports={audit};
