/* -- FAQs -- */
async function loadFAQs() {
  const tb = document.getElementById('faqs-tbody');
  try {
    const r = await apiFetch('/admin/faqs');
    let faqs = await r.json();
    if (!r.ok) throw new Error(faqs.detail || 'Failed');
    faqCache = faqs.sort((a,b)=>(num(b.view_count)-num(a.view_count)) || new Date(b.updated_at||0) - new Date(a.updated_at||0));
    if (!faqCache.length) { tb.innerHTML='<tr><td colspan="8" class="empty">No FAQs yet.</td></tr>'; return; }
    tb.innerHTML = faqCache.map((f,i)=>`<tr>
      <td class="td-mono">${i+1}</td>
      <td style="max-width:240px;" class="td-strong">${esc(f.question)}</td>
      <td style="max-width:260px;font-size:12px;color:var(--muted);">${esc((f.answer||'').substring(0,90))}${(f.answer||'').length>90?'...':''}</td>
      <td><span class="badge badge-gray">${esc(f.category||'General')}</span></td>
      <td><span class="badge badge-gold">${esc(String(f.language||'en').toUpperCase())}</span></td>
      <td><strong>${num(f.view_count).toLocaleString()}</strong></td>
      <td class="td-mono">${dateLabel(f.updated_at||f.created_at)}</td>
      <td><div style="display:flex;gap:6px;flex-wrap:wrap;"><button class="btn btn-sm btn-gold" onclick="editFAQ(${f.id})">Edit</button><button class="btn btn-sm btn-danger" onclick="deleteFAQ(${f.id})">Delete</button></div></td>
    </tr>`).join('');
  } catch(e) { tb.innerHTML=`<tr><td colspan="8"><div class="err-msg">${esc(e.message)}</div></td></tr>`; }
}

function resetFAQForm() {
  editingFaqId = null;
  ['faq-q','faq-a','faq-cat'].forEach(id=>document.getElementById(id).value='');
  document.getElementById('faq-lang').value='en';
}

async function addFAQ() {
  const q=document.getElementById('faq-q').value.trim();
  const a=document.getElementById('faq-a').value.trim();
  if(!q||!a){toast('Question and answer are required.',true);return;}
  try {
    const wasEditing = editingFaqId !== null;
    const payload={question:q,answer:a,category:document.getElementById('faq-cat').value.trim(),language:document.getElementById('faq-lang').value};
    const path=editingFaqId?`/admin/faqs/${editingFaqId}`:'/admin/faqs';
    const method=editingFaqId?'PUT':'POST';
    const r=await apiFetch(path,{method,body:JSON.stringify(payload)});
    const data=await r.json().catch(()=>({}));
    if(!r.ok) throw new Error(data.detail || 'Publish failed.');
    resetFAQForm();
    toast(wasEditing?'FAQ updated.':'FAQ published.');
    loadFAQs();
  } catch(e){toast(e.message,true);}
}

function editFAQ(id) {
  const faq = faqCache.find(item => Number(item.id) === Number(id));
  if(!faq){toast('FAQ not found.',true);return;}
  editingFaqId = Number(id);
  document.getElementById('faq-q').value = faq.question || '';
  document.getElementById('faq-a').value = faq.answer || '';
  document.getElementById('faq-cat').value = faq.category || '';
  document.getElementById('faq-lang').value = faq.language || 'en';
  document.getElementById('faq-q').focus();
  toast('FAQ loaded into the form for editing.');
}

async function deleteFAQ(id) {
  if(!confirm('Delete this FAQ?')) return;
  try {
    const r=await apiFetch(`/admin/faqs/${id}`,{method:'DELETE'});
    if(!r.ok) throw new Error('Failed.');
    toast('FAQ deleted.'); loadFAQs();
  } catch(e){toast(e.message,true);}
}

async function exportFAQs() {
  try {
    const r=await apiFetch('/admin/faqs/export');
    const rows=await r.json();
    if(!r.ok) throw new Error(rows.detail||'Failed to export FAQs.');
    const headers=['id','question','answer','category','language','view_count','created_at','updated_at'];
    const csv=[headers.join(',')].concat((rows||[]).map(row=>headers.map(h=>csvCell(row[h])).join(','))).join('\n');
    const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;
    a.download=`laborlens-faqs-${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast('FAQ export ready.');
  } catch(e){toast(e.message,true);}
}

async function importFAQs(input) {
  const file=input?.files?.[0];
  if(!file) return;
  try{
    const text=await file.text();
    let items=[];
    if(file.name.toLowerCase().endsWith('.json')){
      const parsed=JSON.parse(text);
      items=Array.isArray(parsed)?parsed:(parsed.items||[]);
    }else{
      const lines=text.split(/\r?\n/).filter(Boolean);
      if(lines.length<2) throw new Error('FAQ import file is empty.');
      const headers=lines[0].split(',').map(h=>h.trim().replace(/^"|"$/g,'').toLowerCase());
      items=lines.slice(1).map(line=>{
        const cells=[]; let cur=''; let inQuotes=false;
        for(let i=0;i<line.length;i++){
          const ch=line[i];
          if(ch === '"'){
            if(inQuotes && line[i+1] === '"'){ cur+='"'; i++; }
            else inQuotes=!inQuotes;
          }else if(ch===',' && !inQuotes){
            cells.push(cur); cur='';
          }else cur+=ch;
        }
        cells.push(cur);
        const row={};
        headers.forEach((h,idx)=>row[h]=String(cells[idx]||'').trim());
        return {
          question: row.question || '',
          answer: row.answer || '',
          category: row.category || '',
          language: row.language || 'en'
        };
      }).filter(item=>item.question && item.answer);
    }
    const r=await apiFetch('/admin/faqs/import',{method:'POST',body:JSON.stringify({items})});
    const data=await r.json();
    if(!r.ok) throw new Error(data.detail||'Failed to import FAQs.');
    toast(`Imported ${data.created||0} FAQs.`);
    loadFAQs();
  }catch(e){toast(e.message,true);}
  input.value='';
}

