"""Build the four-page report and a standalone figure booklet from saved data."""
from pathlib import Path
import json,re
from html import unescape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from PIL import Image as PILImage
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'output/reports'; FIG=OUT/'figures'
for n,f in [('TimesNewRoman','times.ttf'),('TimesNewRoman-Bold','timesbd.ttf'),('TimesNewRoman-Italic','timesi.ttf')]:
    pdfmetrics.registerFont(TTFont(n,str(Path('C:/Windows/Fonts')/f)))
pdfmetrics.registerFontFamily('TimesNewRoman',normal='TimesNewRoman',bold='TimesNewRoman-Bold',italic='TimesNewRoman-Italic')
B=ParagraphStyle('Body',fontName='TimesNewRoman',fontSize=12,leading=13.5,spaceAfter=5)
H=ParagraphStyle('Heading',parent=B,fontName='TimesNewRoman-Bold',spaceBefore=6,spaceAfter=4,keepWithNext=True)
E=ParagraphStyle('Equation',parent=B,alignment=1,spaceAfter=5)
C=ParagraphStyle('Caption',parent=B,spaceAfter=7)
story=[];txt=[]
def markup(s):return s.replace('<sub>','<sub size="9" rise="2">').replace('<super>','<super size="9" rise="4">')
def plain(s):return unescape(re.sub('<[^>]+>','',s.replace('<br/>','\n').replace('<sub>','_').replace('<super>','^')))
def p(s,style=B):story.append(Paragraph(markup(s),style));txt.append(plain(s)+'\n')
def h(s):p(s,H)
def eq(s):p(s,E)
def page():story.append(PageBreak());txt.append('\n'+'='*65+'\n')
def fig(name,caption):
    path=FIG/(name+'.png');w,hh=PILImage.open(path).size
    story.append(Image(str(path),width=468,height=468*hh/w));p(caption,C)
def table(rows,widths):
    data=[[Paragraph(markup(str(x)),B) for x in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eef2')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,5)]);txt.extend([' | '.join(map(plain,row)) for row in rows]);txt.append('')

p('Rectangular Microstrip Patch Antenna Optimization',ParagraphStyle('Title',parent=H,alignment=1,spaceBefore=0))
p('EE5606 Artificial Intelligence for Antennas in Wireless Communication',E)
h('1 Aim and analytical model')
p('The aim is to minimize effective antenna volume while maintaining resonance near 2.4 GHz. Part I uses an analytical genetic algorithm (GA); Part II trains neural-network (NN) and Gaussian-process (GP) surrogates and uses particle swarm optimization (PSO). The implementation emphasis is the manually developed GeneticAlgorithm.py, NeuralNetwork.py and PSO.py. GP is a supporting comparison implemented with scikit-learn, rather than the main algorithm-development focus. The NN-GP comparison remains a worksheet requirement [1].')
p('The design is [W, L, h] in mm, with W in [25, 50], L in [25, 45], h in [1, 5] and W/L ≥ 1.2. FR-4 is represented by ε<sub>r</sub> = 4.4 and c = 3 × 10<super>8</super> m/s. The worksheet transmission-line equations are [1]:')
eq('ε<sub>eff</sub> = (ε<sub>r</sub> + 1)/2 + [(ε<sub>r</sub> − 1)/2](1 + 12h/W)<super>−1/2</super>')
eq('ΔL = 0.412h [(ε<sub>eff</sub> + 0.3)(W/h + 0.264)]<br/>/ [(ε<sub>eff</sub> − 0.258)(W/h + 0.8)]')
eq('L<sub>eff</sub> = L + 2ΔL,       f = c/(2L<sub>eff</sub>√ε<sub>eff</sub>),       V = WLh')
p('The effective length is converted to metres for frequency calculation; the public interfaces return GHz and volume is in mm<super>3</super>. The reference [38, 30, 1.6] mm gives 2.357548542 GHz and 1824 mm<super>3</super>.')
fig('01_antenna_and_workflow','Figure 1. Antenna schematic and the two required pipelines. Full-wave simulation is not required; the schematic is not a simulated antenna.')
h('2 Shared objective and acceptance')
eq('J = V/11250 + λ |f<sub>eval</sub> − 2.4|,       λ = 1')
p('Here f<sub>eval</sub> is the analytical or surrogate frequency in GHz. The scale 11250 mm<super>3</super>, penalty λ and acceptance tolerance ±0.01 GHz are design choices. Geometry must be valid and analytical target deviation must be ≤10 MHz; a low surrogate score alone does not establish success.')

page();h('3 Genetic algorithm and neural network implementation')
p('<b>GA.</b> Each chromosome holds [W, L, h]. Tournament selection chooses lower-score parents, arithmetic crossover blends them using one random coefficient, and Gaussian mutation perturbs each coordinate. Geometry repair enforces the bounds and W/L constraint. One distinct elite is retained; duplicates are replaced with fresh feasible designs. The saved study uses population 6, tournament size 2, crossover probability 0.8, mutation probability 0.2 and mutation scale 10% of each coordinate range. It stops below 1% best-score change after at least 100 generations. Ten seeds (42 and 0-8) all satisfy analytical acceptance [2].')
fig('02_ga_convergence','Figure 2. Saved GA histories; the darker curve is the lowest-score run (seed 5). All ten runs completed 100 generations.')
p('<b>Dataset.</b> Uniform sampling rejects W/L &lt; 1.2. The 600 feasible designs are labelled analytically and split into 420 training, 90 validation and 90 test samples, with sampling and split seeds both 1. Scaling uses training-set statistics only; both surrogates share the split.')
fig('03_nn_architecture','Figure 3. Saved NN architecture. Scaling is retained with the weights, so prediction inputs remain in mm and outputs in GHz.')
p('<b>NN.</b> The fully connected 3-32-1 model uses tanh hidden neurons, a linear output and Xavier initialization. For N samples, X has shape (N, 3), hidden activations (32, N), and output (N, 1). Its 161 weights and biases are trained with handwritten forward propagation, mean squared error, backpropagation and parameter updates:')
eq('z<sub>y</sub> = W<sub>2</sub> tanh(W<sub>1</sub>x̃ + b<sub>1</sub>) + b<sub>2</sub>,       θ ← θ − α grad(MSE)')
p('Training uses standardized targets; prediction reverses target scaling to return GHz.')
p('Backpropagation uses tanh derivative 1 − a<super>2</super> and sums each bias gradient over samples. A strong Wolfe line search chooses α with initial trial 1, decrease coefficient 10<super>−4</super> and curvature coefficient 0.1, using bracketing and bisection with a 40-iteration limit. Each update uses all 420 training samples. GPU array operations use CuPy; no automatic differentiation is used.')

page();h('4 Surrogate training and prediction accuracy')
p('The saved NN uses initialization seed 1 and completes 30,000 epochs; the best validation checkpoint is epoch 29,998. Training stops at the epoch limit, without reaching the 10 kHz validation goal. Other stopping conditions are 2000 epochs without improvement or a stalled line search. The best checkpoint is restored before testing [3].')
fig('04_nn_training_error','Figure 4. NN training/validation RMSE and signed test error (prediction minus analytical frequency), from saved histories and predictions. Test size: 90.')
p('<b>GP benchmark.</b> Following GPR.py, scikit-learn fits training-only StandardScaler and an RBF GaussianProcessRegressor: initial length scales [1, 1, 1], alpha = 10<super>−8</super>, normalize_y = True. Marginal-likelihood optimization [5] gives scales [11.3, 2.26, 4.74]. Fitting took 0.183 s. Settings were fixed before testing; the NN checkpoint was reused without retraining [4].')
table([
['<b>Prediction error (MHz)</b>','<b>NN</b>','<b>Library GP</b>'],
['Validation RMSE','0.699299','0.007522'],
['Test RMSE','0.827545','0.009480'],
['Test MAE','0.641820','0.006050'],
['Maximum absolute test error','3.549000','0.057057'],
],[254,107,107])
p('GP has approximately 87 times lower test RMSE. The NN is nevertheless accurate enough for the selected final-design tolerance. Test error measures approximation quality, whereas final analytical target deviation measures whether optimization meets 2.4 GHz; these are different quantities.')
h('5 Particle swarm optimization')
p('PSO.py keeps the surrogate fixed and accepts an objective callback. Forty repaired particles store personal/global bests. Velocity is updated and clipped to 20% of each coordinate range; positions are moved, repaired and scored before updating the best records. Both models use seeds 1-5, inertia 0.7 and cognitive/social coefficients 1.6. Stopping uses &lt;1% improvement after 150 iterations, capped at 1000. Every run completed 150 iterations and 6040 evaluations [4].')
eq('v ← 0.7v + 1.6r<sub>1</sub> × (p<sub>best</sub> − x) + 1.6r<sub>2</sub> × (g<sub>best</sub> − x)')
p('Then x ← repair(x + v); products are elementwise and r<sub>1</sub>, r<sub>2</sub> are uniform random vectors on [0, 1].')

page();h('6 Optimization results and discussion')
fig('06_pso_convergence','Figure 5. CPU NN and library-GP PSO convergence: median across five seeds, with shaded minimum-to-maximum ranges [4].')
table([
['<b>Quantity</b>','<b>GA</b>','<b>NN PSO</b>','<b>GP PSO</b>'],
['W / L (mm)','35.61336 /<br/>29.67780','35.62485 /<br/>29.68737','35.61422 /<br/>29.67851'],
['Volume (mm<super>3</super>)','1056.926','1057.608','1056.977'],
['Analytical f (GHz)','2.399999867','2.399232188','2.399942681'],
['Target deviation (MHz)','0.000133','0.767812','0.057319'],
['Mean search time (s)','Not saved','0.131','1.550'],
],[151,105,106,106])
p('The table uses best-score GA and seed-1 PSO, with five-seed mean times from the new CPU comparison. Success rates are GA 10/10, NN 5/5 and GP 5/5. All designs have h = 1 mm and W/L = 1.2. Both surrogate predictions are ≈2.4 GHz, giving prediction errors of 0.767812 MHz (NN) and 0.057319 MHz (GP).')
p('<b>GPU latency.</b> I initially loaded the trained NN on GPU for PSO. Five-seed searches averaged 2.080 s versus 0.142 s on CPU with identical weights: 14.6× longer [6]. Each particle prediction transfers three inputs to GPU; returning a Python float synchronizes the CPU-visible result. Repeated transfers, kernel launches and synchronization can dominate this tiny network. Transfer delay was not separately profiled, so this remains the likely explanation. CPU inference requires no retraining; batching predictions could reduce transfer overhead.')
p('Library GP is more accurate but slower in this scalar-call CPU comparison. Timings exclude fitting/loading and final verification; NN training and data-generation times are unavailable. Repair also computes analytical RMA quantities internally. These results do not establish an end-to-end speedup over the inexpensive analytical formula.')
h('7 Conclusion')
p('GA, NN and PSO produce analytically valid designs near 2.4 GHz and 1057 mm<super>3</super>. GP supplies an accurate library benchmark. CPU inference reduces NN latency. Global optimality and radiation performance are not established.')
p('<b>Sources:</b> [1] EE5606 worksheet, pp. 1-3. [2-4] results/ga/part1_review.json; results/nn; results/report_library_gp. [5] <link href="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html" color="#245b88">scikit-learn GPR documentation</link>. [6] results/pso_comparison and pso_comparison_cpu.')

def footer(c,d):c.setFont('TimesNewRoman',12);c.drawRightString(540,43,str(d.page))
def build(path,items):
    doc=BaseDocTemplate(str(path),pagesize=(612,792),leftMargin=72,rightMargin=72,topMargin=72,bottomMargin=72,title='EE5606 antenna optimization report',author='')
    frame=Frame(72,72,468,648,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='Report',frames=[frame],onPage=footer))
    doc.build(items)
pdf=OUT/'EE5606_visual_report_draft.pdf'
build(pdf,story)
(OUT/'EE5606_visual_report_draft.txt').write_text('\n'.join(txt),encoding='utf-8')
print('Report pages',len(PdfReader(pdf).pages))
tmp=ROOT/'tmp/report_visual';tmp.mkdir(parents=True,exist_ok=True)
doc=pdfium.PdfDocument(str(pdf))
for i in range(len(doc)):doc[i].render(scale=1.9).to_pil().save(tmp/f'page-{i+1}.png')

captions={
'01_antenna_and_workflow':'Geometry schematic and overall workflow; not to scale.',
'02_ga_convergence':'Source: results/ga/part1_review.json; all ten saved seeds.',
'03_nn_architecture':'Current NeuralNetwork.py and saved 3-32-1 checkpoint.',
'04_nn_training_error':'Saved NN training history and 90 test predictions; signed error in MHz.',
'05_test_parity':'90 held-out designs shared by NN and library GP; dashed line is exact agreement.',
'06_pso_convergence':'results/report_library_gp: median and min-max across seeds 1-5.',
'07_cpu_gpu_latency':'Saved NN searches in pso_comparison and pso_comparison_cpu. Points are individual runs; bars are means. Transfer/synchronization costs were not isolated.',
'08_ga_flowchart':'Current GA flow. Single-generation stopping is a heuristic, not proof of convergence.',
'09_nn_flowchart':'NN stop: validation RMSE target, patience, epoch cap, or failed/zero step; then restore the best checkpoint.',
'10_pso_flowchart':'Current PSO.py; scoring uses the fixed surrogate. The repair wrapper also calculates RMA quantities as a side effect.'}
book=[]
for i,(name,caption) in enumerate(captions.items()):
    if i:book.append(PageBreak())
    title=name[3:].replace('_',' ').title()
    for acronym in ['Ga','Nn','Pso','Cpu','Gpu']:title=title.replace(acronym,acronym.upper())
    book.append(Paragraph(title,H))
    w,hh=PILImage.open(FIG/(name+'.png')).size
    scale=min(468/w,510/hh)
    book.append(Image(str(FIG/(name+'.png')),width=w*scale,height=hh*scale))
    book.append(Spacer(1,10));book.append(Paragraph(caption,B))
build(OUT/'EE5606_figures_and_flowcharts.pdf',book)
