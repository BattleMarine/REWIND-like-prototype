// 브라우저 없이 생성 HTML의 JavaScript를 DOM/canvas mock으로 검사한다.
// 실제 브라우저 CSS/이벤트 호환성 검사를 대체하지 않는다.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');

const target = process.argv[2];
if (!target) throw new Error('Usage: node tests/test_viewer.cjs <viewer.html> [canvas_commands.json]');
const html = fs.readFileSync(target, 'utf8');
const payload = JSON.parse(html.match(/<script id="pose-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
assert.equal(/__POSE_JSON__/.test(html), false);
const nodes = new Map();
const commands = [];
let cursor = [], timerCount = 0;
const ctx = {
  strokeStyle: '', fillStyle: '', lineWidth: 1, dash: [], font: '',
  setTransform() {}, clearRect() {commands.length = 0;},
  setLineDash(d) {this.dash = [...d];}, beginPath() {cursor=[];},
  moveTo(x,y) {cursor.push([x,y]);}, lineTo(x,y) {cursor.push([x,y]);},
  stroke() {commands.push({kind:'line',points:cursor.map(p=>[...p]),color:this.strokeStyle,width:this.lineWidth,dash:this.dash});},
  arc(x,y,r) {cursor=[x,y,r];},
  fill() {commands.push({kind:'circle',circle:[...cursor],color:this.fillStyle});},
  fillText(text,x,y) {commands.push({kind:'text',text,x,y,color:this.fillStyle});},
};
function element(id='') {
  return {id, value:id==='group'?'all':id==='interval'?'200':'0',textContent:'',children:[],disabled:false,
    append(child){this.children.push(child);},replaceChildren(){this.children=[];},
    getBoundingClientRect(){return {width:1000,height:530,left:0,top:0};},
    getContext(){return ctx;},setPointerCapture(){},};
}
function get(id){if(!nodes.has(id))nodes.set(id,element(id));return nodes.get(id);}
get('pose-data').textContent = JSON.stringify(payload);
const timers = new Map();
const context = vm.createContext({document:{getElementById:get,createElement:()=>element()},
  window:{devicePixelRatio:1},ResizeObserver:class {constructor(callback){this.callback=callback;}observe(){this.callback();}},
  setInterval(callback){const id=++timerCount;timers.set(id,callback);return id;},
  clearInterval(id){timers.delete(id);},console});
vm.runInContext(script,context,{timeout:3000});
function evaluate(code){return vm.runInContext(code,context,{timeout:3000});}
function validDrawing(){
  assert.ok(commands.length);
  for(const c of commands)for(const v of c.kind==='circle'?c.circle:c.kind==='line'?c.points.flat():[c.x,c.y])assert.ok(Number.isFinite(v));
}
const summary={data_kind:payload.data_kind,sequence_count:payload.sequences.length,checks:[]};
if(payload.sequences.length && payload.sequences[0].frames.length){
  validDrawing();summary.checks.push('initial_render_finite');
  const s=payload.sequences[0], last=s.frames.length-1;
  const bounds0=evaluate('JSON.stringify(allBounds[0])');
  if(last>0){get('next').onclick();assert.ok(get('frameLabel').textContent.includes(`Frame ${s.frames[1].frame_index}`));get('prev').onclick();}
  get('frameSlider').value=String(last);get('frameSlider').oninput();
  assert.ok(get('frameLabel').textContent.includes(`Frame ${s.frames[last].frame_index}`));
  assert.equal(evaluate('JSON.stringify(allBounds[0])'),bounds0);validDrawing();summary.checks.push('slider_and_fixed_bounds');
  get('frameSlider').value='0';get('frameSlider').oninput();
  for(const id of ['xy','xz','yz','orbit']){get(id).onclick();validDrawing();}
  summary.checks.push('plane_views_finite');
  for(const group of Object.keys(payload.skeleton.groups)){get('group').value=group;get('group').onchange();validDrawing();}
  get('group').value='all';get('group').onchange();
  summary.checks.push('group_selection');
  get('play').onclick();for(let i=0;i<s.frames.length+1;i++)for(const fn of [...timers.values()])fn();
  assert.equal(evaluate('framePosition'),last);assert.equal(timers.size,0);assert.equal(evaluate('sequenceIndex'),0);
  summary.checks.push('playback_stops_at_sequence_end');
  // 두 번째 sequence는 fixture이므로 실제 sample 검증 범위로 집계하지 않는다.
  evaluate('DATA.sequences.push({...DATA.sequences[0], id:"synthetic_reset_fixture"}); allBounds.push(allBounds[0]);');
  get('sequence').value='1';get('sequence').onchange();
  assert.equal(evaluate('framePosition'),0);assert.equal(timers.size,0);assert.equal(evaluate('sequenceIndex'),1);
  summary.checks.push('synthetic_sequence_reset');
  get('sequence').value='0';get('sequence').onchange();
  get('pose').onpointerdown({clientX:0,clientY:0,pointerId:1});
  get('pose').onpointermove({clientX:50,clientY:30});get('pose').onpointerup();validDrawing();
  summary.checks.push('rotation_draw_finite');
  get('orbit').onclick();
  if(process.argv[3])fs.writeFileSync(process.argv[3],JSON.stringify({width:1000,height:530,background:'#0e1923',commands},null,2));
}
console.log(JSON.stringify(summary,null,2));
