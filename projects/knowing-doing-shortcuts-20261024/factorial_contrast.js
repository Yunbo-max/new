// R004: purely numeric factorial contrast utility.
// Input: four arrays of records with keys action, hidden and proxy.
function contrast(cells, outcome) {
 const avg=a=>a.reduce((s,x)=>s+x[outcome],0)/a.length;
 for(const k of ["00","10","01","11"])
   if(!Array.isArray(cells[k])||!cells[k].length) throw Error("missing cell "+k);
 return (avg(cells["11"])-avg(cells["01"]))-(avg(cells["10"])-avg(cells["00"]));
}
function budgetCurve(rows) {
 const groups={};
 for(const r of rows){
   if(r.assigned_budget===undefined||r.actual_tokens===undefined)
     throw Error("unrecorded treatment or consumption");
   const k=String(r.task)+"_"+String(r.assigned_budget);
   (groups[k]??=[]).push(r);
 }
 return Object.entries(groups).map(([key,rs])=>({
    key, n:rs.length,
    success:rs.reduce((s,r)=>s+r.hidden,0)/rs.length,
    mean_actual_tokens:rs.reduce((s,r)=>s+r.actual_tokens,0)/rs.length
 }));
}
module.exports={contrast,budgetCurve};
// Conditional contrast requires randomized cells and a fixed intended task.
// Assigned budget is a treatment; consumption may be post-treatment.
