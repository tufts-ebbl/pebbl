(function(){
var NS="http://www.w3.org/2000/svg",S3=Math.sqrt(3),uid=0,INK="#22282A",GAP="#6B6B67",BLUE="#3E8EDE",SOLID="stroke-dasharray:none;stroke-dashoffset:0;animation:none;transition:none;stroke-linejoin:round;stroke-linecap:round;stroke-miterlimit:10;vector-effect:none",SHARP="stroke-dasharray:none;stroke-dashoffset:0;animation:none;transition:none;stroke-linejoin:miter;stroke-miterlimit:10;vector-effect:none",TEXT="font-family:'Source Sans 3',sans-serif;font-weight:700;font-style:normal;font-stretch:normal;font-variant:normal;letter-spacing:0;word-spacing:0;text-transform:none;dominant-baseline:alphabetic;alignment-baseline:alphabetic;text-anchor:middle;stroke:none;paint-order:normal;fill:"+INK+";";
var PAL=["#F0F0EE","#E4E4E1","#D8D8D4","#CBCBC7","#BEBEB9","#B0B0AB","#A2A29D","#94948F","#DCDAD5","#C9C6C0"];
function mk(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;var t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function el(tag,a,p){var e=document.createElementNS(NS,tag);for(var k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function hexPts(cx,cy,r){var w=r*S3/2;return [[cx,cy-r],[cx+w,cy-r/2],[cx+w,cy+r/2],[cx,cy+r],[cx-w,cy+r/2],[cx-w,cy-r/2]].map(function(p){return p[0].toFixed(1)+","+p[1].toFixed(1);}).join(" ");}
function inHex(x,y,cx,cy,r){var dx=Math.abs(x-cx),dy=Math.abs(y-cy);return dx<=r*S3/2&&dy<=r-dx/S3;}
function pebble(g,x,y,rad,Rn,line){
 var e=0.08+Rn()*0.3,rx=rad*(1+e),ry=rad*(1-e*0.55),col=PAL[Math.floor(Rn()*PAL.length)];
 var pg=el("g",{transform:"translate("+x.toFixed(1)+","+y.toFixed(1)+") rotate("+(Rn()*180).toFixed(0)+")"},g);
 el("ellipse",{rx:rx.toFixed(1),ry:ry.toFixed(1),fill:col,stroke:"#000","stroke-opacity":"0.2","stroke-width":line,style:SOLID},pg);
 el("ellipse",{cx:(-rx*0.22).toFixed(1),cy:(-ry*0.3).toFixed(1),rx:(rx*0.55).toFixed(1),ry:(ry*0.33).toFixed(1),fill:"#FFF","fill-opacity":"0.28",style:"stroke:none"},pg);
 if(rad>line*9&&Rn()<0.2)el("path",{d:"M "+(-rx*0.7).toFixed(1)+" "+(ry*0.1).toFixed(1)+" Q 0 "+(-ry*0.25).toFixed(1)+" "+(rx*0.72).toFixed(1)+" "+(ry*0.2).toFixed(1),fill:"none",stroke:"#FFF","stroke-opacity":"0.6","stroke-width":line,style:SOLID},pg);
}
function inkMap(tx,W,fs){
 var cs=getComputedStyle(tx),font=cs.fontWeight+" "+fs+"px "+cs.fontFamily;
 var H=Math.ceil(fs*1.7),base=Math.ceil(fs*1.15),c=document.createElement("canvas");c.width=W;c.height=H;
 var x=c.getContext("2d");x.font=font;x.textAlign="left";x.textBaseline="alphabetic";x.fillStyle="#000";
 var x0=tx.getStartPositionOfChar(0).x,ratio=tx.getComputedTextLength()/x.measureText("pebbl").width;
 x.setTransform(ratio,0,0,1,x0*(1-ratio),0);x.fillText("pebbl",x0,base);
 var d=x.getImageData(0,0,W,H).data,sky=new Array(W),flo=new Array(W);
 for(var i=0;i<W;i++){sky[i]=Infinity;flo[i]=-Infinity;for(var j=0;j<H;j++){if(d[(j*W+i)*4+3]>40){sky[i]=j-base;break;}}if(isFinite(sky[i]))for(var j2=H-1;j2>=0;j2--){if(d[(j2*W+i)*4+3]>40){flo[i]=j2-base;break;}}}
 return {sky:sky,flo:flo,asc:x.measureText("b").actualBoundingBoxAscent};
}
function quadPts(x0,y0,xc,yc,x1,y1,n){var o=[];for(var k=0;k<=n;k++){var t=k/n,u=1-t;o.push([u*u*x0+2*u*t*xc+t*t*x1,u*u*y0+2*u*t*yc+t*t*y1]);}return o;}
window.PEBBL=function(host,o){
 var id="pb"+Math.floor(Math.random()*1e9)+"_"+(uid++),W=o.W,H=o.H,cx=W/2,cy=H/2,R=1000,ri=R-o.border,wi=ri*S3/2,E=wi+80,half=o.tw/2,clr=half+o.gap;
 var top=cy-ri/2,bot=cy+ri/2,b=top+o.halo/2,Lb=bot-o.halo/2,tip=cy-ri+o.tipDrop;
 var svg=el("svg",{viewBox:"0 0 "+W+" "+H,width:"100%",role:"img"});host.appendChild(svg);
 var cp=el("clipPath",{id:id},el("defs",{},svg));el("polygon",{points:hexPts(cx,cy,ri+2)},cp);
 var tx=el("text",{x:cx,y:cy,"text-anchor":"middle",style:TEXT},svg);tx.textContent="pebbl";
 var fs=o.fs,ty,Sx,Sy,Pl=cx-390,Pc=cx-315,Pr=cx-240,Tl,Tc,Tr,tl,tc,tr,pl,pc,pr;
 var seg=function(x0,y0,x1,y1){var n=Math.ceil(Math.hypot(x1-x0,y1-y0)/4),a=[];for(var k=0;k<=n;k++)a.push([x0+(x1-x0)*k/n,y0+(y1-y0)*k/n]);return a;};
 for(var it=0;it<14;it++){
  tx.style.fontSize=fs+"px";tx.setAttribute("y",cy);
  var m=inkMap(tx,W,fs);ty=b+o.textTop+m.asc;
  var sky=m.sky.map(function(v){return v+ty;}),flo=m.flo.map(function(v){return v+ty;});
  var ascTop=Math.min.apply(null,sky),runs=[],cur=null;
  for(var i=0;i<W;i++){var on=sky[i]<=ascTop+fs*0.03;if(on&&!cur){cur={a:i,b:i};}else if(on){cur.b=i;}else if(cur){runs.push(cur);cur=null;}}
  if(cur)runs.push(cur);
  Sx=runs.length>=2?(runs[0].b+runs[1].a)/2:cx+fs*0.25;
  var hits=function(pts){for(var k=0;k<pts.length;k++){var px=pts[k][0],py=pts[k][1];for(var xx=Math.floor(px-clr);xx<=Math.ceil(px+clr);xx++){if(xx<0||xx>=W||!isFinite(sky[xx]))continue;var dz=Math.sqrt(Math.max(0,clr*clr-(xx-px)*(xx-px)));if(sky[xx]<py+dz&&flo[xx]>py-dz)return true;}}return false;};
  Sy=ascTop+o.dipBelowTop;while(Sy>b+40&&hits(seg(cx,tip,Sx,Sy).concat(seg(Sx,Sy,Sx+55,b))))Sy-=3;
  Tl=Sx+120;Tc=Sx+240;Tr=Sx+360;tl=2*cx-Tr;tc=2*cx-Tc;tr=2*cx-Tl;pl=2*cx-Pr;pc=2*cx-Pc;pr=2*cx-Pl;
  var lowPts=seg(cx-E,Lb,tl,Lb).concat(quadPts(tl,Lb,tc,Lb-190,tr,Lb,60)).concat(seg(tr,Lb,pl,Lb)).concat(quadPts(pl,Lb,pc,Lb-110,pr,Lb,40)).concat(seg(pr,Lb,cx+E,Lb));
  if(!hits(lowPts))break;
  fs-=8;
 }
 function loY(x){function bump(x0,x1,A){var t=(x-x0)/(x1-x0);return Lb-2*t*(1-t)*A;}
  if(x>tl&&x<tr)return bump(tl,tr,190);if(x>pl&&x<pr)return bump(pl,pr,110);return Lb;}
 svg.removeChild(tx);
 if(o.bleed)el("polygon",{points:hexPts(cx,cy,R+o.bleed),fill:INK,"class":"guides",style:"stroke:none"},svg);
 el("polygon",{points:hexPts(cx,cy,ri+2),fill:GAP,style:"stroke:none"},svg);
 var g=el("g",{"clip-path":"url(#"+id+")"},svg),Rn=mk(o.seed),placed=[];
 for(var t=0;t<o.tries;t++){
  var rad=o.minR+(o.maxR-o.minR)*Math.pow(Rn(),2)*(t<o.tries*0.2?1:0.6);
  var x=cx+(Rn()*2-1)*(wi+rad),y=cy+(Rn()*2-1)*(ri+rad);
  if(!inHex(x,y,cx,cy,ri+rad*0.9))continue;
  if(y>top+rad*0.1&&y<loY(x)+o.halo*0.2)continue;
  var clash=false;for(var i2=0;i2<placed.length;i2++){var p=placed[i2],dx=x-p.x,dy=y-p.y,mm=(rad+p.r)*0.9;if(dx*dx+dy*dy<mm*mm){clash=true;break;}}
  if(clash)continue;placed.push({x:x,y:y,r:rad});pebble(g,x,y,rad,Rn,o.line);
 }
 var f=function(v){return v.toFixed(1);};
 var q="M "+f(cx-E)+" "+f(b)+" L "+f(Pl)+" "+f(b)+" Q "+f(Pc)+" "+f(b-110)+" "+f(Pr)+" "+f(b)+" L "+f(cx-120)+" "+f(b)+" L "+f(cx-75)+" "+f(b+70)+" L "+f(cx)+" "+f(tip)+" L "+f(Sx)+" "+f(Sy)+" L "+f(Sx+55)+" "+f(b)+" L "+f(Tl)+" "+f(b)+" Q "+f(Tc)+" "+f(b-190)+" "+f(Tr)+" "+f(b)+" L "+f(cx+E)+" "+f(b);
 var lo="M "+f(cx-E)+" "+f(Lb)+" L "+f(tl)+" "+f(Lb)+" Q "+f(tc)+" "+f(Lb-190)+" "+f(tr)+" "+f(Lb)+" L "+f(pl)+" "+f(Lb)+" Q "+f(pc)+" "+f(Lb-110)+" "+f(pr)+" "+f(Lb)+" L "+f(cx+E)+" "+f(Lb);
 var loRev="L "+f(cx+E)+" "+f(Lb)+" L "+f(pr)+" "+f(Lb)+" Q "+f(pc)+" "+f(Lb-110)+" "+f(pl)+" "+f(Lb)+" L "+f(tr)+" "+f(Lb)+" Q "+f(tc)+" "+f(Lb-190)+" "+f(tl)+" "+f(Lb)+" L "+f(cx-E)+" "+f(Lb)+" Z";
 var tg=el("g",{"clip-path":"url(#"+id+")"},svg);
 el("path",{d:q+" "+loRev,fill:"#FFFFFF",stroke:"none",style:"stroke:none"},tg);
 [q,lo].forEach(function(d){el("path",{d:d,fill:"none",stroke:"#FFFFFF","stroke-width":o.halo,"stroke-linejoin":"round","stroke-linecap":"round",style:SOLID},tg);});
 [q,lo].forEach(function(d){el("path",{d:d,fill:"none",stroke:BLUE,"stroke-width":o.tw,"stroke-linejoin":"round","stroke-linecap":"round",style:SOLID},tg);});
 el("polygon",{points:hexPts(cx,cy,R-o.border/2),fill:"none",stroke:INK,"stroke-width":o.border,"stroke-linejoin":"miter",style:SHARP},svg);
 tx.setAttribute("y",f(ty));tx.style.fontSize=fs+"px";svg.appendChild(tx);
 if(o.bleed){var gd=el("g",{"class":"guides"},svg);
  el("polygon",{points:hexPts(cx,cy,R),fill:"none",stroke:"#E0218A","stroke-width":8,style:"stroke-dasharray:30 20"},gd);
  el("polygon",{points:hexPts(cx,cy,R-125),fill:"none",stroke:"#00A6D6","stroke-width":8,style:"stroke-dasharray:30 20"},gd);
  el("polygon",{points:hexPts(cx,cy,R+o.bleed),fill:"none",stroke:"#999","stroke-width":8,style:"stroke-dasharray:10 14"},gd);
  [top,bot].forEach(function(y){el("line",{x1:cx-wi,y1:y,x2:cx+wi,y2:y,stroke:"#00A6D6","stroke-width":6,style:"stroke-dasharray:12 12"},gd);});}
 return {fs:fs,Sx:Sx,Sy:Sy};
};
})();
