const $ = (sel)=>document.querySelector(sel);
const sleep = (ms)=>new Promise(r=>setTimeout(r,ms));

// 1) Upload
$('#btnUpload').addEventListener('click', async ()=>{
  const f = $('#fileInput').files;
  if(!f || !f.length){ alert('Choose at least one image'); return; }
  const fd = new FormData();
  for(const x of f) fd.append('files', x);
  $('#btnUpload').disabled = true;
  $('#uploadMsg').textContent = 'Uploading...';
  try{
    const r = await fetch('/api/upload', {method:'POST', body:fd});
    const j = await r.json();
    $('#uploadMsg').textContent = j.message || 'Done';
  }catch(e){
    $('#uploadMsg').textContent = 'Upload failed: '+e.message;
  }finally{
    $('#btnUpload').disabled = false;
  }
});

// 2) Run Smart Resizer
$('#btnRunResizer').addEventListener('click', async ()=>{
  $('#resizerLogs').textContent = 'Running...';
  $('#btnRunResizer').disabled = true;
  try{
    const r = await fetch('/api/run-resizer', {method:'POST'});
    const j = await r.json();
    $('#resizerLogs').textContent = j.logs || '(no output)';
  }catch(e){
    $('#resizerLogs').textContent = 'Error: '+e.message;
  }finally{
    $('#btnRunResizer').disabled = false;
  }
});

// 3) Results grid
async function refreshCopies(){
  $('#copiesGrid').innerHTML = '<div class="muted">Loading...</div>';
  try{
    const r = await fetch('/api/list-copies');
    const j = await r.json();
    if(!j.items || !j.items.length){
      $('#copiesGrid').innerHTML = '<div class="muted">No items. Run Smart Resizer first.</div>';
      return;
    }
    const frag = document.createDocumentFragment();
    for(const it of j.items){
      const div = document.createElement('div');
      div.className = 'result-card';
      div.innerHTML = `
        <img src="${it.url256}" alt="256"/>
        <img src="${it.url512}" alt="512"/>
        <div class="meta">
          <div class="title" title="${it.name}">${it.name}</div>
          <div class="sub" title="256: ${it.rel256}\n512: ${it.rel512}">256: ${it.rel256} • 512: ${it.rel512}</div>
        </div>
        <div class="actions">
          <button class="btn-danger btnDelPair" data-r256="${it.rel256}" data-r512="${it.rel512}" onclick="deletePair('${it.rel256}', '${it.rel512}')" role="button">Delete</button>
        </div>`;
      frag.appendChild(div);
    }
    $('#copiesGrid').innerHTML = '';
    $('#copiesGrid').appendChild(frag);
  }catch(e){
    $('#copiesGrid').innerHTML = '<div class="muted">Error: '+e.message+'</div>';
  }
}
$('#btnRefreshCopies').addEventListener('click', refreshCopies);

// 4) Upscale + Compare
async function pollMetrics(){
  const r = await fetch('/api/get-metrics');
  const j = await r.json();
  const body = $('#metricsBody'); body.innerHTML = '';
  if(!j.rows || !j.rows.length){
    body.innerHTML = '<tr><td colspan="4" class="muted">No metrics yet.</td></tr>';
    return;
  }
  for(const row of j.rows){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.filename}</td><td>${row.mse}</td><td>${row.psnr}</td><td>${row.ssim}</td>`;
    body.appendChild(tr);
  }
}

async function runBatch(){
  const mode = $('#sourceSel').value;
  const limit = $('#limitSel').value;
  $('#batchLogs').textContent = '';
  $('#etaText').textContent = 'Running batch inference...';
  $('#progressBar').style.width = '10%';
  $('#btnRunBatch').disabled = true;
  try{
    const r = await fetch('/api/run-upscale-compare', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({mode, limit: Number(limit)})
    });
    const j = await r.json();
    $('#batchLogs').textContent = j.logs || '(no logs)';
    $('#progressBar').style.width = '100%';
    $('#etaText').textContent = 'Done';
    await pollMetrics();
  }catch(e){
    $('#batchLogs').textContent = 'Error: '+e.message;
    $('#etaText').textContent = 'Error';
  }finally{
    $('#btnRunBatch').disabled = false;
    setTimeout(()=>$('#progressBar').style.width='0%', 1500);
  }
}
$('#btnRunBatch').addEventListener('click', runBatch);
$('#btnRefreshMetrics').addEventListener('click', pollMetrics);

// download zip of outputs
$('#btnZipOutputs').addEventListener('click', async ()=>{
  const r = await fetch('/api/zip-upscale-outputs', {method:'POST'});
  const j = await r.json();
  if(j.url) window.location = j.url;
});

// Initial paint
refreshCopies();
pollMetrics();


async function listUploads(){
  const r = await fetch('/api/list-uploads');
  const j = await r.json();
  const box = document.getElementById('uploadsList');
  box.innerHTML = '';
  if(!j.items || !j.items.length){
    box.innerHTML = '<div class="muted">No uploads yet.</div>';
    return;
  }
  for(const it of j.items){
    const row = document.createElement('div');
    row.className = 'list-item';
    row.innerHTML = `<span title="${it.path}">${it.name}</span>
                     <button class="btn-danger btnDelUpload" data-name="${it.name}">Delete</button>`;
    box.appendChild(row);
  }
}

document.getElementById('btnToggleUploads').addEventListener('click', async ()=>{
  const pnl = document.getElementById('uploadsPanel');
  const vis = pnl.style.display !== 'none';
  pnl.style.display = vis ? 'none' : 'block';
  if(!vis) await listUploads();
});

document.getElementById('uploadsList').addEventListener('click', async (e)=>{
  const btn = e.target.closest('.btnDelUpload');
  if(!btn) return;
  if(!confirm('Delete this upload file?')) return;
  const name = btn.dataset.name;
  const r = await fetch('/api/delete-upload', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name})
  });
  const j = await r.json();
  if(!j.ok){ alert('Delete failed: ' + (j.error||'unknown')); }
  await listUploads();
});


window.deletePair = async function(rel256, rel512){
  try{
    const payload = {rel256, rel512};
    const r = await fetch('/api/delete-resizer-pair', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if(!j.ok){
      alert('Delete failed: ' + (j.error || 'unknown'));
    }
  }catch(e){
    alert('Delete error: ' + e.message);
  }
  await refreshCopies();
};

document.getElementById('copiesGrid').addEventListener('click', async (e)=>{
  const btn = e.target.closest('.btnDelPair');
  if(!btn) return;
  const rel256 = btn.dataset.r256;
  const rel512 = btn.dataset.r512;
  await window.deletePair(rel256, rel512);
});
