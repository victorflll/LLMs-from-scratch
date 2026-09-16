// Run: node 23-09/tests/test_gpt_arch.cjs — numeric checks, no browser dependencies.
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const source=readFileSync(__dirname+'/../web/v1/gpt_arch.js','utf8').split('let state=')[0];
const m=vm.runInNewContext(source+';({forward,blockForward,blocks,normRow,gelu,tokW,posW,outW,V,MAX_T,vocab,guidedIds,nextGuided})');
const close=(a,b,eps=1e-10)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
const input=[0,1,2,3],a=m.forward(input),future=m.forward([0,1,2,7]);
for(const block of [a.b1,a.b2]){
 for(const head of block.attention)head.forEach((row,i)=>{close(row.reduce((s,x)=>s+x,0),1);row.forEach((v,j)=>{assert.ok(v>=0);if(j>i)assert.equal(v,0);});});
 for(const row of block.r2){assert.equal(row.length,8);assert.ok(row.every(Number.isFinite));}
}
// Causality must survive both complete blocks, not just the softmax matrix.
for(let i=0;i<3;i++)for(let j=0;j<m.V;j++)close(a.logits[i][j],future.logits[i][j]);
const extended=m.forward([...input,4]);
for(let i=0;i<4;i++)for(let j=0;j<m.V;j++)close(a.logits[i][j],extended.logits[i][j]);
// Removing the mask must expose the prefix to changes in future tokens.
const u=m.forward(input,false),uf=m.forward([0,1,2,7],false);
assert.ok(u.logits[0].some((x,j)=>Math.abs(x-uf.logits[0][j])>1e-5));
for(let len=1;len<=m.MAX_T;len++){
 const result=m.forward(Array.from({length:len},(_,i)=>i%m.V));
 assert.equal(result.logits.length,len);
 for(const row of result.probs){close(row.reduce((s,x)=>s+x,0),1);assert.ok(row.every(x=>x>0&&x<1));}
}
// Normalization and activation have expected boundary behavior.
assert.ok(m.normRow([3,3,3,3]).every(x=>x===0));
const n=m.normRow([-2,-1,0,1,2,3,4,5]);close(n.reduce((s,x)=>s+x,0),0);assert.ok(n.reduce((s,x)=>s+x*x,0)/8>.999);
close(m.gelu(0),0);assert.ok(m.gelu(-1)<0);assert.ok(Math.abs(m.gelu(3)-3)<.005);
// Count actual stored parameters; LayerNorm gamma/beta are implicit ones/zeros.
const count=x=>Array.isArray(x)?x.reduce((s,v)=>s+count(v),0):typeof x==='object'?Object.values(x).reduce((s,v)=>s+count(v),0):1;
assert.equal(count(m.tokW)+count(m.posW)+count(m.outW)+count(m.blocks)+5*16,2000);
console.log('OK: causal invariance, future leakage comparison, shapes 1–12 tokens, probability sums, LayerNorm, GELU and 2,000 parameters.');

const guided=[0,1,2,3];
while(m.nextGuided(guided)!==null){guided.push(m.nextGuided(guided));const result=m.forward(guided);assert.equal(result.logits.length,guided.length);assert.ok(result.logits.flat().every(Number.isFinite));}
assert.equal(guided.map(id=>m.vocab[id]).join(''),'Hello, I am a GPT built from scratch');
assert.equal(m.nextGuided([0,1,2,3,7]),null);
console.log('OK: guided continuation completes in five insertions and stops; invalid prefixes rejected.');
