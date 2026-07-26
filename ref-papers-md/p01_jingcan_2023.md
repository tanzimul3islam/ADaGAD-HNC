| Normality |             | Learning-based |     |             | Graph    | Anomaly  | Detection  | via |
| --------- | ----------- | -------------- | --- | ----------- | -------- | -------- | ---------- | --- |
|           |             | Multi-Scale    |     | Contrastive |          | Learning |            |     |
|           | JingcanDuan |                |     |             | PeiZhang |          | SiweiWang∗ |     |
jingcan_duan@163.com zhangpei@nudt.edu.cn wangsiwei13@nudt.edu.cn
NationalUniversityofDefense NationalUniversityofDefense IntelligentGameandDecisionLab
|     | Technology           |     |     |                      | Technology |     | Beijing,China |     |
| --- | -------------------- | --- | --- | -------------------- | ---------- | --- | ------------- | --- |
|     | Changsha,Hunan,China |     |     | Changsha,Hunan,China |            |     |               |     |
|     | JingtaoHu            |     |     |                      | HuJin      |     | JiaxinZhang   |     |
hujingtao17@nudt.edu.cn jinhu@nudt.edu.cn zhangjx001206@163.com
NationalUniversityofDefense NationalUniversityofDefense NationalUniversityofDefense
|     | Technology |     |     |     | Technology |     | Technology |     |
| --- | ---------- | --- | --- | --- | ---------- | --- | ---------- | --- |
Changsha,Hunan,China Changsha,Hunan,China Changsha,Hunan,China
|     |     | HaifangZhou∗                |            |     |                             | XinwangLiu∗            |     |     |
| --- | --- | --------------------------- | ---------- | --- | --------------------------- | ---------------------- | --- | --- |
|     |     | haifang_zhou@nudt.edu.cn    |            |     |                             | xinwangliu@nudt.edu.cn |     |     |
|     |     | NationalUniversityofDefense |            |     | NationalUniversityofDefense |                        |     |     |
|     |     |                             | Technology |     |                             | Technology             |     |     |
|     |     | Changsha,Hunan,China        |            |     |                             | Changsha,Hunan,China   |     |     |
ABSTRACT algorithmimprovesthedetectionperformance(upto5.89%AUC
gain)comparedwiththestate-of-the-artmethods.Thesourcecode
Graphanomalydetection(GAD)hasattractedincreasingattention
inmachinelearninganddatamining.Recentworkshavemainly isreleasedathttps://github.com/FelixDJC/NLGAD.
focusedonhowtocapturericherinformationtoimprovethequality
CCSCONCEPTS
ofnodeembeddingsforGAD.Despitetheirsignificantadvances
indetectionperformance,thereisstillarelativedearthofresearch •Computingmethodologies→Anomalydetection;•Mathe-
onthepropertiesofthetask.GADaimstodiscerntheanomalies maticsofcomputing→Graphalgorithms.
| that deviate                                           | from most nodes. | However, | the model | is prone | to       |     |     |     |
| ------------------------------------------------------ | ---------------- | -------- | --------- | -------- | -------- | --- | --- | --- |
| learnthepatternofnormalsampleswhichmakeupthemajorityof |                  |          |           |          | KEYWORDS |     |     |     |
samples.Meanwhile,anomaliescanbeeasilydetectedwhentheir
GraphAnomalyDetection,NormalityLearning,Multi-ScaleCon-
| behaviorsdifferfromnormality.Therefore,theperformancecan |     |     |     |     | trastiveLearning |     |     |     |
| -------------------------------------------------------- | --- | --- | --- | --- | ---------------- | --- | --- | --- |
befurtherimprovedbyenhancingtheabilitytolearnthenormal
ACMReferenceFormat:
pattern.Tothisend,weproposeanormalitylearning-basedGAD
JingcanDuan,PeiZhang,SiweiWang,JingtaoHu,HuJin,JiaxinZhang,
frameworkviamulti-scalecontrastivelearningnetworks(NLGAD
HaifangZhou,andXinwangLiu.2023.NormalityLearning-basedGraph
forabbreviation).Specifically,wefirstinitializethemodelwith
AnomalyDetectionviaMulti-ScaleContrastiveLearning.InProceedingsof
thecontrastivenetworksondifferentscales.Toprovidesufficient the31stACMInternationalConferenceonMultimedia(MM’23),October29–
andreliablenormalnodesfornormalitylearning,wedesignan November3,2023,Ottawa,ON,Canada.ACM,NewYork,NY,USA,10pages.
effectivehybridstrategyfornormalityselection.Finally,themodel https://doi.org/10.1145/3581783.3612064
isrefinedwiththeonlyinputofreliablenormalnodesandlearnsa
moreaccurateestimateofnormalitysothatanomalousnodescan
|     |     |     |     |     | 1   | INTRODUCTION |     |     |
| --- | --- | --- | --- | --- | --- | ------------ | --- | --- |
bemoreeasilydistinguished.Eventually,extensiveexperimentson
Recently,graphanomalydetection(GAD)hasbecomeanincreasing
sixbenchmarkgraphdatasetsdemonstratetheeffectivenessofour
applicationofgraph-basedmachinelearningforresearchers[17,
normalitylearning-basedschemeonGAD.Notably,theproposed
25,37].Well-establishedgraphanomalydetectionalgorithmscan
∗Correspondingauthors. effectivelydetectanomaloussampleswhosebehaviorobviously
straysfromthemajorityofnodesinagraph.Owingtoitsexcellent
Permissiontomakedigitalorhardcopiesofallorpartofthisworkforpersonalor
|     |     |     |     |     | performance | in preventing | real-world harmful | situations, GAD |
| --- | --- | --- | --- | --- | ----------- | ------------- | ------------------ | --------------- |
classroomuseisgrantedwithoutfeeprovidedthatcopiesarenotmadeordistributed
forprofitorcommercialadvantageandthatcopiesbearthisnoticeandthefullcitation hasbeenusedinawiderangeofareas,includingsocialnetwork
onthefirstpage.Copyrightsforcomponentsofthisworkownedbyothersthanthe anomaly[9,38],socialspamdetection[30],medicaldomain[8],
author(s)mustbehonored.Abstractingwithcreditispermitted.Tocopyotherwise,or networkintrusiondetection[44],etc.StructuredgraphdatainGAD
republish,topostonserversortoredistributetolists,requirespriorspecificpermission
and/orafee.Requestpermissionsfrompermissions@acm.org. containsbothnodefeatureinformationandnetworkstructurein-
MM’23,October29–November3,2023,Ottawa,ON,Canada.
formation.Themismatchbetweensuchtwotypesofinformation
©2023Copyrightheldbytheowner/author(s).PublicationrightslicensedtoACM.
generatestwotypicalanomalies,i.e.contextualandstructuralanom-
ACMISBN979-8-4007-0108-5/23/10...$15.00
https://doi.org/10.1145/3581783.3612064 alies[4,16,18].Theformeristhenodesthataredissimilartotheir
7502

MM’23,October29–November3,2023,Ottawa,ON,Canada. JingcanDuanetal.
improveCoLAbyaddingnode-nodecontrast,reconstructionloss,
andglobalinformation,respectively.Overall,theyfocusonimprov-
ingthequalityofnoderepresentationsthroughdiversetypesof
informationforGAD.
However, existing works ignore the characteristics of graph
anomalydetection.AsindicatedinFigure1,wedrawthedistribu-
tiondiagramsoftheanomalyscorescomputedbyANEMONE[11]
ondatasetsCoraandDBLP.InSubfigure1aand1c,ATrepresents
thatallnodesareusedasthetraininginput,whileNTmeansonly
thenormalnodesaretreatedasthetraininginputinSubfigure1b
and 1d. It is worth noting that the anomaly score is calculated
fromallnodesintheinferencephase.Theintuitiveobservation
(a)Cora-AT. (b)Cora-NT. isthatthedistributionsofnormalnodes(blue)arefurtheraway
fromthedistributionsofanomalies(red)inNTthaninAT.Usu-
ally,thesmalleroverlapregionbetweenthesetwodistributions
meansthatthetrainedmodelhasastrongeranomalydetection
capability.Sincenormalnodesmakeupthemajorityofsamplesin
agraph,GADmodelsaremorepronetolearnthepatternofnormal
samples.ItissummarizedasNormalityLearninginGAD.Nodes
thatdeviatefromthispatternareusuallyrecognizedasanomalous
ones.Itisareasonableapproachtoboostdetectionperformance
byfurtherenhancingtheabilitytolearnthenormalpattern.The
phenomenoninFigure1demonstratesthatifonlynormalnodesare
utilizedinthetrainingphase,theabilityofthemodeltolearnthe
normalpatternwillbetremendouslyenhanced.Conversely,mixing
(c)DBLP-AT. (d)DBLP-NT. anomalousandnormalnodeswillharmthenormalitylearning
andweakenthedetectionperformance.Basedontheseanalyses,
Figure1:Anomalyscoredistributiondiagrams.Thesesub-
howtoimprovedetectionperformanceutilizingnormalitylearning
figuresarethedistributionsofanomalyscorescomputed
underanunsupervisedparadigminGADisstillanissueworth
byANEMONEfornodesonCoraandDBLP.Inthemodel
investigating.
trainingphase,(a)and(c)useallnodesasthetraininginput
To this end, we propose a Normality Learning-based Graph
(AT).Differently,(b)and(d)onlyinputthenormalnodesfor
AnomalyDetectionframeworkviamulti-scalecontrastivelearn-
training(NT).
ingnetworkstermedNLGAD.Theentireframeworkisshownin
neighbors.Thelatterindicatesthatsomenodeshavedissimilar Figure2.Toinitializethemodel,wefirstconstructthecontrastive
featuresbutareunusuallycloselyconnected. networkscontainingsubgraph-node(SN)andnode-node(NN)con-
Earlierresearchesimproveanomalydetectionperformancethrough trasts, which can fuse multi-scale of anomalous information in
featureengineering,whichcannothandlehigh-dimensionalnode graph.Themodelistrainedusingatwo-phasescheme.Inthenor-
featuresingraph.Inrecentyears,theemergenceofgraphneural malityselectionphase,weleverageallnodesastheinputtotrain
networks(e.g.GCN[12])hasbroughthopeforsolvingtheappeal- themodel.Inthisprocess,wedeviseahybridselectionstrategy
ingproblem[19–23].ResearchersinstinctivelyintroduceittoGAD. thatassignsnormalpseudo-labelstosufficientnodes.Tobespecific,
DOMINANT [4] designs an ingenious GCN-based autoencoder. weevaluatetheanomalydegreeforeachnodeateachstepand
Thenitcomparestheoriginalfeatureandadjacencymatriceswith thenaddthehigh-confidenceestimatestothenormalitypoolwith
reconstructedmatricesafterGCN.Anomalousnodeshavegreater theimprovementofmodelcapability.Attheendofthisphase,we
variationsthannormalones.DOMINANTachievesagreatcom- synthetically compute the anomaly estimate for each node and
parativeadvantageoverearliermethods.Nevertheless,GCNisa assignnormalpseudo-labelstothelowestpartofthenodes.Inthe
low-passfilterinessence[1]andwillsmooththeanomalieswith normalitylearningphase,themodelisretrainedwiththeinputof
theirneighbors,whichdoesharmtotheGADtask.Toovercomethis selectednormalnodesbasedonthenormalitypool.Bytakingad-
inherentdefect,researchersredesignspecificgraphfilterswhich vantageofnormallearning,thenetworkwillbefurtherrefinedfor
obtainanomalousinformationfromdifferentfrequencies[3,32].As thetask.Finally,wecalculatethefinalanomalyscoresforallnodes.
withtheothersemi-supervisedapproaches[13,14],theystillsuffer Furthermore,itisremarkablethatNLGADdoesnotrequiresuper-
fromtheexpensivelabel-collectionprogress.Withoutthereliance visioninformationfromtheground-truthlabelsthroughoutthe
onground-truthlabelsofknownanomalies,thepioneeringwork trainingphase.Insummary,ourcontributionscanbesummarized
CoLA[18]firstintroducesthegraphcontrastivelearningpattern inthefollowings:
toGAD.Itdigslocalfeatureandstructureinformationthrough
subgraph-nodecontrast.Normalnodesusedtobesimilartotheir • WeempiricallyinvestigatethedifferentimpactsofATand
neighborhoodsinpositivepairsanddissimilarinnegativepairs, NTonGADandexplainthesuperiorityofnormalitylearning
whileanomaliesbehavedifferently.Subsequentworks[11,41,42] inthistask.
7503

NormalityLearning-basedGraphAnomalyDetectionviaMulti-ScaleContrastiveLearning MM’23,October29–November3,2023,Ottawa,ON,Canada.
• Weproposeanormalitylearning-basedgraphanomalyde- Table1:Tableofmainsymbols.
tectionschemeonattributednetworks,withoutanymanual
| annotation. | Notations |     | Definitions |     |     |     |
| ----------- | --------- | --- | ----------- | --- | --- | --- |
• Anovelhybridnormalityselectionstrategyisdesignedto
|     | G   |     | Anundirectedattributedgraph |     |     |     |
| --- | --- | --- | --------------------------- | --- | --- | --- |
pickoutsufficientandreliablenormalnodestoprovidesup-
|                           | 𝑣      |     | The𝑖-thnodeofG     |     |     |     |
| ------------------------- | ------ | --- | ------------------ | --- | --- | --- |
| portfornormalitylearning. | 𝑖      |     |                    |     |     |     |
|                           | A∈R𝑛×𝑛 |     | AdjacencymatrixofG |     |     |     |
•
| ExperimentsdemonstratethenotableadvantageofNLGAD | D∈R𝑛×𝑛 |     |     |     |     |     |
| ------------------------------------------------ | ------ | --- | --- | --- | --- | --- |
DegreematrixofA
| againstcurrentgraphanomalydetectioncompetitors,which | X∈R𝑛×𝑑 |     |     |     |     |     |
| ---------------------------------------------------- | ------ | --- | --- | --- | --- | --- |
FeaturematrixofG
indicatestheeffectivenessofnormalitylearning.Andtheab-
|                                                        | 𝒙𝑖 ∈R1×𝑑    |     | Featurevectorof𝑣 | that𝒙𝒊         | ∈X  |        |
| ------------------------------------------------------ | ----------- | --- | ---------------- | -------------- | --- | ------ |
| lationstudiesfurthervalidatethenecessityofthenormality |             |     |                  | 𝑖              |     |        |
|                                                        | H(ℓ) ∈R𝑛×𝑑′ |     |                  |                |     | ℓ-th   |
| selectionstrategy.                                     |             |     | Subgraph hidden  | representation |     | of the |
layer
|     | 𝒉(ℓ) ∈R1×𝑑′  |     | Nodehiddenrepresentationoftheℓ-thlayer |     |     |     |
| --- | ------------ | --- | -------------------------------------- | --- | --- | --- |
|     | W(ℓ) ∈R𝑑′×𝑑′ |     | Networkparametersoftheℓ-thlayer        |     |     |     |
𝑦 ∈{0,1}
|     | 𝑖   |     | LabelofinstancepairinBCEloss |     |     |     |
| --- | --- | --- | ---------------------------- | --- | --- | --- |
2 RELATEDWORK
|     | 𝑠𝑐𝑜𝑟𝑒(𝑣 | 𝑖)  | Finalanomalyscoreof𝑣 |     | 𝑖   |     |
| --- | ------- | --- | -------------------- | --- | --- | --- |
Inthefollowingsection,weprovideabriefreviewofgraphanomaly
detectionviadeeplearning.Graphneuralnetworks(GNNs)[12,15,
| 27,34]havesignificantadvantagesincapturingdeeperinformation | 4 METHOD |     |     |     |     |     |
| ----------------------------------------------------------- | -------- | --- | --- | --- | --- | --- |
ofgraphdatasets.BasedonGNNs,[36,43]extendthetraditional
Inthefollowingsection,wepresenttheproposedframework,NL-
anomalydetectionalgorithmone-classSVMtothegraphdatasets.
GAD.Wefirstsamplesubgraphsaroundnodesandformsubgraph-
Graphgenerativemodelshavealsoachievednotableimprovement. nodecontrast.Themodelalsoleveragesnode-nodecontrasttocap-
DOMINANT[4]firstadoptsaGCN-basedautoencoderingraph turethenode-levelanomalousinformation.Afterthat,themodel
anomalydetectionandtreatsthenodeswithlargerreconstruction
willbeinitialized.Then,wetrainthemodelthroughtwophases.
errorsasanomalies.BasedonDOMINANT,researchersboostthe
Duringthenormalityselectionphase,themodelistrainedwith
detectionperformancebyfusingthecommunityinformationof
|     | all nodes | and estimates | the anomaly | degree | for each | node. We |
| --- | --------- | ------------- | ----------- | ------ | -------- | -------- |
eachnode[24].[3,32]enhancethecapabilityofhigh-passgraph addhigh-confidentestimatesintothenormalitypoolbyatailored
filtersandrelievetheover-smoothingphenomenoncarriedbyGCN. dynamicstrategyandretainthelowestpartofnodes.Duringthe
Self-supervisedlearningisanimportantbranchofunsupervised
secondphase,werefinethemodelwiththeselectednormalnodes
learning.HCM[10]regardsthepredictivehopsofthetargetnode
andthencalculatethefinalanomalyscoresforallnodes.Finally,
anditsneighborhoodsasitsanomalousdegree.Thelargerthenum-
weperformcomplexityanalysisonNLGAD.
berofpredictivehops,thehigherprobabilityofbeinganomalies.
Thepioneer[18]leveragesthegraphcontrastivelearningpattern 4.1 ModelInitialization
forthefirsttimeandcapturesanomaliesfromnode-subgraphcom-
Themulti-scalegraphcontrastivepatternhasbeenprovenvalidfor
parison.Ithasbecomeastrongbaselineingraphanomalydetection.
Basedonit,ANEMONE[11]proposesanewnode-nodecontrastive GAD[11].Thus,webuildagraphcontrastivenetworkunderthe
pattern.GRADATE[6]optimizesthesubgraphembeddingsinGAD multi-scalestrategyasthebackboneofourmodel.Tobespecific,
wefirstadoptrandomwalktosamplesubgraphsaroundnodes.
byaddingasubgraph-subgraphcontrast.Differently,SL-GAD[42]
Then,weconstructthesubgraph-nodeandnode-nodecontrasts
combinestheadvantagesofreconstructionandgraphcontrastive
separately.Theformercontrastcapturesanomalousinformation
learningparadigms.Sub-CR[41]capturesmoreanomalousinfor-
mationbyutilizinganewlygeneratedglobalview.[5]focusesmore fromnodeneighborhoods.Thesecondonefurtherdigsoutthe
ondetectingstructuralanomalies. node-levelanomalies.Basedonbothcontrasts,themodelwillbe
initializedbyinformationfromnodesandtheirneighborhoods.
|     | 4.1.1 Subgraph-Node |     | (SN) contrast. | Firstly, | we sample | the sub- |
| --- | ------------------- | --- | -------------- | -------- | --------- | -------- |
3 PROBLEMDEFINITION
graphsaroundthenodestocreatesubgraph-nodecontrasts.The
Inthissection,weformallydefinethemainproblem.Themain featureandstructureinformationfromthesubgraphscanrepresent
notationsaresummarizedinTable1. thelocalneighborhoodinformationofthenodes,whichisuseful
Problemdefinition(GraphAnomalyDetection).Anundirected fordiscerninganomalies.Insubgraph-nodecontrast,thetarget
| attributedgraphG=(V,E)iscomposedof:(1)asetofnodesV, | node𝑣 |     |     |     |     |     |
| --------------------------------------------------- | ----- | --- | --- | --- | --- | --- |
𝑖 formspositivepairswiththesubgraphswhereitislocated
where|V|=𝑛;(2)asetofedgesE,where|E|=𝑚.Specifically,the
andformsnegativepairswiththesubgraphswhereanothernode
graphisformalizedasitsfeaturematrix𝑋 andadjacencymatrix islocated.Inspiredby[29,33],weadoptrandomwalkwithrestart
𝐴.Inaddition,theadjacencymatrixA𝑖𝑗 = 1representsthereis (RWR)asthesubgraphsamplingmethod.
an edge between node𝑣 and𝑣 𝑗; otherwise, A𝑖𝑗 = 0. In graph WeutilizeaGCNlayertolearnthesubgraphrepresentationsin
𝑖
anomalydetection,themodelistrainedtolearnaspecificfunction thelatentspace.Itisworthnotingthatthefeaturesofthetarget
| 𝑓 (·)thatcancalculatetheanomalyscore𝑠𝑐𝑜𝑟𝑒(𝑣 | node𝑣 |     |     |     |     |     |
| ------------------------------------------- | ----- | --- | --- | --- | --- | --- |
𝑖)forthenode 𝑖 inthesubgraphhavebeenmaskedto0,whichistoavoid
𝑣 𝑖.Thelargertheanomalyscore,themorelikelyitisthatthenode thesmoothingofanomalousinformation.Thesubgraphrepresen-
| isanomalous. | tationsinthehiddenlayercanbedenotedas: |     |     |     |     |     |
| ------------ | -------------------------------------- | --- | --- | --- | --- | --- |
7504

MM’23,October29–November3,2023,Ottawa,ON,Canada. JingcanDuanetal.
Figure2:TheoverviewframeworkofNLGAD.Itconsistsofthreemodules:(1)ModelInitialization:Weinitializethemodelby
multi-scalecontrastivenetworkswhichhavesubgraph-node(SN)andnode-node(NN)contrasts;(2)NormalitySelection:We
trainthemodelwithallnodesandperformanomalydegreeestimationateachstep.Then,weperformahybridstrategyand
assignthenormalpseudo-labelstoreliablenodes;(3)NormalityLearning:Weinputthereliablenormalnodesintothemodel
andre-trainit.Finally,therefinedmodelisusedtocalculatethefinalanomalyscoreforeachnode.
weutilizeanewMLPtomapthenodefeaturesintothesamespace.
|     |     | (cid:18) |     | (cid:19) |     |     |     |     |     |     |     |
| --- | --- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
(ℓ+1) −1 − 1 (ℓ) W(ℓ) And𝒆ˆ𝑖 isthetargetnode’sfinalembedding.Thenwecanmeasure
|        | H =𝜎                                       | D(cid:101)𝑖 2A(cid:101)𝑖D(cid:101)𝑖 | 2 H | ,   | (1) |                   |     |                                  |     |     |     |
| ------ | ------------------------------------------ | ----------------------------------- | --- | --- | --- | ----------------- | --- | -------------------------------- | --- | --- | --- |
|        | 𝑖                                          |                                     | 𝑖   |     |     | thesimilarityof𝒖𝑖 |     | and𝒆ˆ𝑖 throughaBilinearfunction. |     |     |     |
| (ℓ+1)  | (ℓ)                                        |                                     |     |     |     |                   |     |                                  |     |     |     |
| whereH | andH arethehiddenrepresentationofthe(ℓ+1)- |                                     |     |     |     |                   |     |                                  |     |     |     |
𝑖 𝑖 4.1.3 Lossfunction. Insubgraph-nodecontrast,thetargetnode
thandℓ-thlayer,A(cid:101)𝑖 isthesubgraphadjacencymatrixwithself- tendstobesimilartothesubgraphinthepositivepair,i.e.𝑠 𝑖 =1.
loop,D(cid:101)𝑖 isthedegreematrixofA(cid:101)𝑖,W(ℓ) indicatesthenetwork Differently,ittendstobedissimilartothesubgraphinthenegative
parameters.And𝜎(·)istheactivationfunctionReLUhere. pair,i.e.𝑠 =0.Itisnaturaltousethebinarycross-entropy(BCE)
𝑖
WeadoptaReadoutfunctiontoobtainthefinalsubgraphrepre- losstotrainthenetworks:
| sentation𝒛𝑖.Inpractice,anaveragefunctionisusedtoachieveit. |     |     |     |     |     |     |     | 𝑛                    |           |     |      |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------------------- | --------- | --- | ---- |
|                                                            |     |     |     |     |     |     |     | ∑︁ (𝑦 𝑖log(𝑠 𝑖)+(1−𝑦 | 𝑖)log(1−𝑠 |     | 𝑖)), |
| Thefinalrepresentationisdefinedas:                         |     |     |     |     |     |     | L𝑆𝑁 | =−                   |           |     | (5)  |
|                                                            |     |     | 𝑛𝑖  |     |     |     |     | 𝑖=1                  |           |     |      |
1
|     | =𝑅𝑒𝑎𝑑𝑜𝑢𝑡(Z𝑖)= |     | ∑︁    | .   |     | where𝑦 |                                                 |     |     |     |     |
| --- | ------------- | --- | ----- | --- | --- | ------ | ----------------------------------------------- | --- | --- | --- | --- |
|     | 𝒛𝑖            |     | (Z𝑖)𝑗 |     | (2) |        | 𝑖 isequalto1inpositivepairsandisequalto0innega- |     |     |     |     |
𝑛 𝑖
|     |     |     | 𝑗=1 |     |     | tivepairs.Thenode-nodecontrastlossfunctionL𝑁𝑁 |     |     |     |     | canalsobe |
| --- | --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- | --- | --------- |
definedlikeL𝑁𝑁.
Inthemeantime,weutilizeaMLPtomapthetargetnodefeatures
Afterthat,wefusethediverseanomalousinformationwitha
tothesamelatentspaceasthesubgraph:
comprehensivelossfunction:
|     | (ℓ+1) | (cid:16) | (ℓ) (cid:17) |     |     |     |     |                    |     |     |     |
| --- | ----- | -------- | ------------ | --- | --- | --- | --- | ------------------ | --- | --- | --- |
|     | 𝒉𝑖    | =𝜎 𝒉𝑖    | W(ℓ) ,       |     | (3) |     |     |                    |     |     |     |
|     |       |          |              |     |     |     |     | L=𝛼·L𝑆𝑁 +(1−𝛼)·L𝑁𝑁 |     | ,   | (6) |
(ℓ)
whereW 𝑖 issharedwithGCNinEq.(1).Thefinalembeddingof where𝛼 (0,1)isatrade-offparametertobalancetheimportance
| thetargetnode𝑣 |     |     |     |     |     |     | ∈   |     |     |     |     |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
𝑖 isdenotedas𝒆𝑖.
betweentwocontrasts.
| Thesimilarity𝑠 | 𝑖ofthetargetnode𝑣 |     | 𝑖anditssubgraphisdirectly |     |     |     |     |     |     |     |     |
| -------------- | ----------------- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
relatedtoitsanomalydegrees[18].WeuseaBilinearfunctionto
|     |     |     |     |     |     | 4.2 | NormalitySelection |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- |
measureit:
𝑠 =𝐵𝑖𝑙𝑖𝑛𝑒𝑎𝑟(𝒛𝑖 ,𝒆𝑖). Previousunsupervisedapproachesinotherfields[7,26,40]retain
|     | 𝑖   |     |     |     | (4) |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
onlyasmallnumberofnodeswithhigh-confidentpseudo-labels.
| 4.1.2 Node-Node(NN)contrast. |     | Togetdiverseinformation,we |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
However,normalitylearninginGADrequiresasufficientnumber
leveragethenode-nodecontrasttocapturenode-levelanomalies.
|     |     |     |     |     |     | of normal | nodes | to train the model | and | learn a more | accurate |
| --- | --- | --- | --- | --- | --- | --------- | ----- | ------------------ | --- | ------------ | -------- |
Thetargetnodeembeddingaggregatedfromtheothernodesinthe estimateforthenormalpattern.Therefore,thenormalityselection
subgraphformsapositivepairwiththesamenodeafterMLP,and strategyshouldconsidernotonlythereliabilitybutalsothenumber
formsanegativepairwithanothernodeafterMLP.Samewiththe ofnormalnodes.Inthisphase,weproposeanewhybridstrategy
subgraph-nodecontrast,thetargetnodefeaturesinthesubgraph
toconductnormalityselectionbyinputtingallthenodesintothe
willbemasked.WeadoptanewGCNtoobtainthesubgraph’s
trainingmodel.Specifically,wefirstemployadynamicstrategy
|     | ′(ℓ) | ′(ℓ) | ′(ℓ) |     |     |     |     |     |     |     |     |
| --- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
representationH 𝑖 ,while𝒉𝑖 =H 𝑖 [1,:]isthehiddenrepre- thatconductsanomalydegreeestimationforeachnodeateach
sentationof𝑣 𝑖.𝒖𝑖 isthefinalembeddingoftargetnode.Similarly, stepandgraduallyaddshigh-confidentestimatestothenormality
7505

NormalityLearning-basedGraphAnomalyDetectionviaMulti-ScaleContrastiveLearning MM’23,October29–November3,2023,Ottawa,ON,Canada.
pool.Apercentstrategyisthenusedtoassignnormalpseudo- 4.2.3 Percentstrategy. Asinglestrategycannotassignhigh-confident
labelstothelowestportionofnodes. pseudo-labelstonodes.Wedesignapercentstrategytolabelnodes
whileleavingenoughnodesasnormalnodes.Tomitigatetheran-
| 4.2.1 | Anomalydegreeestimation. |     | Ateachstep,weestimatearough |     |     |     |     |     |     |     |     |
| ----- | ------------------------ | --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
domnessofdifferentstepsandimprovethereliabilityoftheanom-
anomalydegreeforeachnode.Insubgraph-nodecontrast,anormal
|     |     |     |     |     |     | aly degree | for each | node, | we average | their multi-step | anomaly |
| --- | --- | --- | --- | --- | --- | ---------- | -------- | ----- | ---------- | ---------------- | ------- |
nodetendstobesimilartothesubgraphinthepositivepairand
estimatesinthenormalitypool:
dissimilartotheoneinthenegativepair.Conversely,ananomalous
n o d e is d i ss im i la r f r o m t h e su b gr a p h s in b o t h po s it i v e an d ne g a ti v e 𝑚𝑖
|     |     |     |     |     |     |     |     |     | 1 ∑︁ |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- |
pa i r s. T h e re fo r e , t h e a n o m al y d e g r ee o f t h e ta r g e t n o de c a n b e 𝐸(𝑣 𝑖)′ = 𝐸(𝑣 𝑖)𝑞 , (10)
𝑚
| representedas: |     |     |     |     |     |     |     |     | 𝑖 𝑞=1 |     |     |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- |
𝑝 w h e re 𝑚 𝑖 i s th e t im es o f 𝑣 𝑖 ’ s a n o m a l y d e gr e e es ti m a t es a d d e d to
|     |     | 𝐸 (𝑣 𝑖)=𝑠 | 𝑛 −𝑠 , |     | (7) |     |     |     |     |     |     |
| --- | --- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
𝑆𝑁 𝑖 𝑖 th e n or m a l ity p o o l.A ft e r t h a t , w e s o r tt h e e s tim a t es i n as c e n d in g
𝑝
where𝑠 and𝑠 𝑛 representthesimilarityofpositiveandnegative orderandreturntheindexvector,
|     | 𝑖 𝑖 |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
pai rs , re s p e c t iv ely . =𝑎𝑟𝑔𝑠𝑜𝑟𝑡(cid:8)𝐸(𝑣 )′,𝐸(𝑣 )′,...,𝐸(𝑣 𝑛)′(cid:9),
|        |                  |                                      |     |     |     |     | 𝒃   |     | 1   | 2   | (11) |
| ------ | ---------------- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
| L ik e | w i s e , w e ca | nalsocomputetheanomalydegreeestimate |     |     |     |     |     |     |     |     |      |
𝐸 𝑁𝑁 (𝑣 𝑖) in node-node contrast. By integrating the anomalous where𝒃 istheindexvector.Then,thenodeswiththelowest𝐾
informationfromtwocontrasts,wecalculatetheanomalydegree percentestimateswillbeassignednormalpseudo-labels.
| estimate𝐸(𝑣 | 𝑖)forthetargetnode𝑣 |               | 𝑖:  |        |     |     |                   |     |     |     |     |
| ----------- | ------------------- | ------------- | --- | ------ | --- | --- | ----------------- | --- | --- | --- | --- |
|             | 𝐸(𝑣 𝑖)=𝛼·𝐸          | (𝑣 𝑖)+(1−𝛼)·𝐸 |     | (𝑣 𝑖), | (8) | 4.3 | NormalityLearning |     |     |     |     |
|             |                     | 𝑆𝑁            |     | 𝑁𝑁     |     |     |                   |     |     |     |     |
where𝛼 issharedwithEq.(6). 4.3.1 Refinetraining. Afterobtainingreliablenormalnodes,we
refinethemodelaccordingtoanormalitylearning-basedscheme.Its
4.2.2 Dynamicstrategy. Thedetectioncapabilityofthemodelwill purposeistoboostthemodel’sabilitytolearnthepatternofnormal
increaseasthenetworkistrained.Therefore,theanomalydegree nodes.Thenwecanobtainamoreaccurateestimateofnormality.
estimatescalculatedateachsteparenotfullyreliable.Weadopt Afterthat,intheinferencephase,theanomalyscoredistributionof
adynamicstrategytoretainthemostreliablepartoftheresults.
normalitywillbefurtherawayfromthedistributionofanomalies.
AsshowninSubfigure.1aand1c,lotsofnormalnodesareusually
Inthisphase,weutilizethenodeswithnormalpseudo-labelsto
erroneouslyassignedhigheranomalyscoresbyGADmodels,which retrainthemodelwith𝑇 𝑟 epochs.
confusesthemwithanomalousnodes.Conversely,thelowestpart
|     |     |     |     |     |     | 4.3.2 | Final anomaly | score | calculation. | Finally, in | the inference |
| --- | --- | --- | --- | --- | --- | ----- | ------------- | ----- | ------------ | ----------- | ------------- |
oftheanomalyscoresisrarelymisclassifiedasanomalousnodes.
Thissuggeststhatlowervaluesinthedistributionofanomalyscores phase,weusethemodeltocomputethefinalanomalyscorefor
haveahigherconfidencelevel.Therefore,wegraduallystorethe eachnode.OnedetectionwithRWRwillmissmuchsemanticin-
nodeswiththelowestproportionofanomalydegreeestimatesin formation.Hence,weemployamulti-rounddetectionstrategy.In
thenormalitypool.Weutilizeaspeedfunction𝑝(𝑗)tospecifythe eachround,wecalculatethe𝑠𝑐𝑜𝑟𝑒(𝑣 𝑖)for𝑣 𝑖 viaEq.(7)(8).Inspired
numberofstoredestimatesatstep𝑗.Forthecharacteristicsofthe by[11],wecomputethefinalanomalyscore:
task,wearguethatthisvitalfunctionmustsatisfythefollowing
c o nd it io n s : ( 1) 𝑝 ( 𝑗 ) s t ar t s w i t h a s m a l l e r va l u e . It c o r r e s p o n d s t o 𝑟
|     |     |     |     |     |     |     |     | 1∑︁ | 𝑖)(𝑘), |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- |
th e in it ia l l o w d et e c ti o n p e r fo r m a n c eo f t h e m o d e l. A s t a r t v a l u e t h a t 𝑠𝑐𝑜𝑟𝑒(𝑣 𝑖)𝑚𝑒𝑎𝑛 = 𝑠𝑐𝑜𝑟𝑒(𝑣
𝑟
𝑘=1
istoolargewilladdmoreunreliableestimatestothenormalitypool.
| 𝑝 (𝑗 |     |     |     |     |     |     |     | (cid:118)(cid:117)(cid:116) |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- | --------------------------- | --- | --- | --- |
( 2 ) ) i s m on o t o n ic a ll y a n d s l o w l y in cr e a s i n g w i t h r e s p e c t 𝑟 (12)
|     |     |     |     |     |     | 𝑠𝑐𝑜𝑟𝑒(𝑣 |     | 1∑︁ | (cid:16) 𝑠𝑐𝑜𝑟𝑒(𝑣 | 𝑖)(𝑘) −𝑠𝑐𝑜𝑟𝑒(𝑣 | (cid:17)2 , |
| --- | --- | --- | --- | --- | --- | ------- | --- | --- | ---------------- | -------------- | ----------- |
t o 𝑗 .D e te c t ion pe r f o rm a n c ei n c re a s e ss l ow ly a s t h e m o d e l is t r a in e d . 𝑖)𝑠𝑡𝑑 = 𝑖)𝑚𝑒𝑎𝑛
𝑟
| Itisnaturaltomake𝑝(𝑗)continuetogrow.Meanwhile,asmall |     |     |     |     |     |     |     | 𝑘=1 |     |     |     |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
growthratecanavoidaddingmarginalanomalyestimates,ensuring 𝑠𝑐𝑜𝑟𝑒(𝑣 𝑖)=𝑠𝑐𝑜𝑟𝑒(𝑣 𝑖)𝑚𝑒𝑎𝑛+𝑠𝑐𝑜𝑟𝑒(𝑣 𝑖)𝑠𝑡𝑑 ,
thatthenormalitypoolretainsreliableestimates.(3)𝑝(𝑗)
ends
|     |     |     |     |     |     | where𝑟 | isthenumberofanomalydetectionround,and𝑠𝑐𝑜𝑟𝑒(𝑣 |     |     |     | 𝑖)  |
| --- | --- | --- | --- | --- | --- | ------ | --------------------------------------------- | --- | --- | --- | --- |
withalargervalue.Thisworkswellwiththepercentstrategy
|     |     |     |     |     |     | isthefinalanomalyscorefor𝑣 |     |     | 𝑖.  |     |     |
| --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | --- | --- | --- |
andcouldassignnormalpseudo-labelstoasmanyreliablenodes
aspossible.Inpractice,weadoptafunctionfromthetan(·)family
|     |     |     |     |     |     | 4.4 | ComplexityAnalysis |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- |
toachieveit:
Weanalyzethetimecomplexityofeachcomponentinourmethod.
(cid:18)𝜋 𝑗 (cid:19)
𝑝(𝑗)=𝑛·tan · . (9) ThetimecomplexityofeachRWRsubgraphsamplingforallnodes
|     |     |     | 𝑇   |     |     | isO(𝑐𝛿𝑛),where𝑐isthenumberofnodeswithinthesubgraphs, |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- |
4 𝑠
Theanomalydegreeestimatesofallnodesateachsteparenor- 𝛿 is the mean degree of the network, and 𝑛 is the number of
malized to [0,1].𝑝(𝑗) represents the number of node anomaly nodes in the graph. The time complexity of the GCN network
estimatesaddedtothenormalitypoolatstep𝑗,𝑛isthenumberof forallnodesisO(cid:0)(cid:0)𝐾𝑞𝑑+𝐾𝑐𝑑2(cid:1)𝑛(cid:1) ,where𝐾 isthenumberoflay-
nodesinthegraph,and𝑇 ers,𝑑 isthedimensionofnodeattributesinhiddenspace,and𝑞
𝑠 isthenumberoftrainingstepsinthe
normalityselectionmodule.Itisworthnotingthattheotherfunc- isthenumberofedgesinsubgraphs.Forthetrainingandinfer-
tionscanalsobeappliedto𝑝(𝑗).Weleavethistechnicalextension encephase,theoveralltimecomplexityoftheproposedmodel
|                |     |     |     |     |     | isO(cid:0)𝑛(cid:0)𝑐𝛿+𝐾𝑞𝑑+𝐾𝑐𝑑2(cid:1)(𝑇 |     |     | +𝑟)(cid:1) |              |              |
| -------------- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | ---------- | ------------ | ------------ |
| forfuturework. |     |     |     |     |     |                                        |     |     | 𝑠 +𝑇 𝑟     | ,where𝑇 𝑠 +𝑇 | 𝑟 isthetotal |
7506

MM’23,October29–November3,2023,Ottawa,ON,Canada. JingcanDuanetal.
numberofepochsinthetrainingphase,𝑟istheroundofdetections twomethodsaretraditionalshallowmethodsandtheotherswork
intheinferencephase. ondeepneuralnetworks.
5 EXPERIMENT 5.1.5 ModelSettings. Tobalanceperformanceandcomplexity,we
setthesizeofsubgraphsto4.Boththefeaturesofnodesandsub-
Inthissection,weperformextensiveexperimentsonsixwidely-
graphsaremappedto64dimensions.Thelearningrateofthemodel
used benchmark datasets toverify the effectiveness ofNLGAD.
isfixedat0.001.Weperform700epochsoftotaltrainingonCora,
Firstly,weintroducethedetailsofexperimentalsettingsandre-
UAI2010,andCiteSeer,900epochsonDBLP,Citation,andACM.
sults.Then,weshowtheablationstudiesofthehybridstrategy
fornormalityselectionandnormalitylearning.Finally,weconduct
5.2 ExperimentalResults
sensibilityanalysesofhyper-parametersintheframework.
Table3showstheresultsofperformanceestimatesbasedonAUC
5.1 ExperimentalSettings valuesfornineapproaches.Simultaneously,wecanobservethe
5.1.1 Datasets. Table2listssixcommonlyusedbenchmarkdatasets ROCcurvesinFigure3,whichintuitivelydemonstratethemodel
inGAD.ThedatasetsincludeCora[31],UAI2010[35],CiteSeer[31], performancebytheirunder-lineareas.Throughcomparison,we
DBLP, Citation, and ACM [39]. UAI2010 is a graph dataset for candrawthefollowingconclusions:
communitydetection.Theothersarecitationdatasets,whichcon- • NLGADoutperformstheothermodelsonalldatasets.
taincitingrelationsofpublicationsorcoauthorrelationshipsof
ItachievessignificantAUCgainsof 0.94%,5.89%,2.09%,
researchers.
0.63%,3.40%,and2.05%onCora,UAI2010,CiteSeer,DBLP,
Citation,andACM,separately.Theseresultsverifytheeffec-
Table2:Thestatisticsofdatasets.
tivenessofNLGADanditsgreatimprovementindetection
performance.Especially,NLGADachieves1.63%-5.89%gains
Datasets Nodes Edges Features Anomalies onAUCvaluesagainstANEMONE.
• Mostdeepneuralnetwork-basedmethodsworkbetter
Cora 2708 5429 1433 150
thanshallowmethods.Amongthem,modelsadoptingthe
UAI2010 3067 28311 4973 150
contrastivelearningparadigmoutperformtheothers.These
CiteSeer 3327 4732 3703 150
indicatethatdeepmethods,especiallycontrastivelearning
DBLP 5484 8117 6775 300
methods,canprocesshigh-dimensionalgraphdatasetsbet-
Citation 8935 15098 6775 450
terthantraditionalmethodsanddigvaluablefeaturesand
ACM 9360 15556 6775 450
structureinformationfromthem.
5.1.2 AnomalyInjection. Ourmethodaimsatdetectingthetwo
5.3 AblationStudy
typicaltypesofanomaliesinthegraph.Toverifyitseffectiveness,
5.3.1 HybridStrategyforNormalitySelection. Toconfirmtheef-
wefollow[4]toinjectsuchtwoanomaliesintotheoriginalgraph:
fectivenessoftheproposedhybridstrategyfornormalityselection,
• Contextualanomaliesareinjectedbyperturbingthefea-
weimplementablationstudyexperiments.Insteadofthedynamic
turevalues.Inpractice,werandomlyselectanode𝑣 𝑖 and
another𝑛′(fixedto50)nodes.Andweexchange𝑣 𝑖’sfeature strategy,weretaintheanomalydegreeestimatesofallnodesateach
valueswiththemostdissimilarnodeoutofthe𝑛′nodes. stepandaveragethembeforethepercentstrategy(termedNLGAD-
AAS).Then,weconductanotherexperiment,onlyretainingthelast
• Structuralanomaliesaregeneratedbyperturbingthetopo-
step’sestimatesbeforethepercentstrategy(termedNLGAD-OLS).
logicalstructureofthegraph.Tobespecific,werandomly
select𝑚′nodesandmakethemfullyconnected.Thenumber Table4showsthatNLGADworksbetterthanNLGAD-AASand
𝑚′isusually15. NLGAD-OLSonalldatasets,whichconfirmstheeffectivenessof
thestrategy.Thisfurtherindicatesthatthedetectionabilityofthe
Werepeattheabovetwooperations,injectingthesamenumber
initialmodelisrelativelyweak,andweshouldgraduallyintroduce
ofcontextualandstructuralanomaliesintotheoriginalgraphs.
themostreliableestimatestothenormalitypool.Thedynamic
Thetotalnumberofanomaliesineachdatasetisshowninthelast
strategymeetsitsrequirementsforspeedcontrol,whilethepercent
columnofTable2.
strategyfurtherguaranteesthequalityofpseudo-labels.
5.1.3 Metric. Intheexperiments,weemploythewidely-usedmethod
5.3.2 NormalityLearning. Besides,weperformanotherablation
AUCasthemetrictoevaluatethemodel.AUCistheunder-line
studyexperimenttoverifythevalidityofnormalitylearning.We
areaoftheROCcurve,whichcanrepresenttheprobabilityofselect-
train the backbone networks with the same epochs as the nor-
inganomaliesratherthannormalityintopanomalyscoresamples.
malityselectionmodulebutdonotperformnormalityselection
ThelargertheAUCvalue,thebetterthemodelperformance.
andnormalitylearning(termedNLGAD-OSP).Inthemeantime,
5.1.4 Baselines. Inthispart,wecompareNEBULAwitheightwell- wetrainthemodelusingthesamenumberoftotalepochsasNL-
knownGADmethods,includingLOF[2],ANOMALOUS[28],DOM- GADwithoutnormalityselectionandnormalitylearning(termed
INANT[4],CoLA[18],ANEMONE[11],SL-GAD[42],HCM[10], NLGAD-SNP).Theformerisusedtoverifytheeffectivenessof
andSub-CR[41].FollowingCoLA,theuseddatasetsinANOMA- normalitylearning.Thelatterexcludesinterferencefromthenum-
LOUSarereducedto30byPCA.Itisworthnotingthatthefirst beroftotaltrainingepochs.Table5demonstratesthatNLGAD
7507

NormalityLearning-basedGraphAnomalyDetectionviaMulti-ScaleContrastiveLearning MM’23,October29–November3,2023,Ottawa,ON,Canada.
Table3:PerformancecomparisonforAUC.Theboldandunderlinedvaluesindicatethebestandrunner-upresults,respectively.
| Category |     | Methods   |     | Cora   |     | UAI2010 |     | CiteSeer | DBLP   |     | Citation |     | ACM    |
| -------- | --- | --------- | --- | ------ | --- | ------- | --- | -------- | ------ | --- | -------- | --- | ------ |
|          |     | LOF(2000) |     | 0.3538 |     | 0.7052  |     | 0.3484   | 0.2694 |     | 0.3059   |     | 0.2843 |
Shallow
|      |         | ANOMALOUS(2018) |     | 0.6688 |     | 0.7144      |     | 0.6581 | 0.6728 |     | 0.6356      |     | 0.5894 |
| ---- | ------- | --------------- | --- | ------ | --- | ----------- | --- | ------ | ------ | --- | ----------- | --- | ------ |
|      |         | DOMINANT(2019)  |     | 0.8929 |     | 0.7698      |     | 0.8718 | 0.8034 |     | 0.7748      |     | 0.8152 |
|      |         | CoLA(2021)      |     | 0.9065 |     | 0.7949      |     | 0.8863 | 0.7824 |     | 0.7296      |     | 0.8127 |
|      |         | ANEMONE(2021)   |     | 0.9122 |     | 0.8731      |     | 0.9227 | 0.8322 |     | 0.8028      |     | 0.8300 |
| Deep |         | SL-GAD(2021)    |     | 0.9192 |     | 0.8454      |     | 0.9177 | 0.8461 |     | 0.8095      |     | 0.8450 |
|      |         | HCM(2021)       |     | 0.6276 |     | 0.5210      |     | 0.6502 | 0.5572 |     | 0.5414      |     | 0.5507 |
|      |         | Sub-CR(2022)    |     | 0.9133 |     | 0.8571      |     | 0.9248 | 0.8061 |     | 0.7903      |     | 0.8428 |
|      |         | NLGAD(Proposed) |     | 0.9286 |     | 0.9320      |     | 0.9457 | 0.8524 |     | 0.8435      |     | 0.8655 |
|      | (a)Cora |                 |     |        |     | (b)UAI2010  |     |        |        |     | (c)CiteSeer |     |        |
|      | (d)DBLP |                 |     |        |     | (e)Citation |     |        |        |     | (f)ACM      |     |        |
Figure3:ROCcurvescomparisononsixbenchmarkdatasets. Theareaunderthecurveislarger,theanomalydetection
performanceisbetter.Theblackdottedlinesarethe“randomline”,indicatingtheperformanceunderrandomguessing.
Table4:Ablationstudyofnormalityselectionstrategyw.r.t. Table5:Ablationstudyofnormalitylearningw.r.t.AUC.
AUC.
|     |     |     |     |     |     |     |     | Cora | UAI2010 | CiteSeer | DBLP | Citation | ACM |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ------- | -------- | ---- | -------- | --- |
Cora UAI2010 CiteSeer DBLP Citation ACM NLGAD-OSP 0.9143 0.9143 0.9340 0.8407 0.8233 0.8612
|                  |     |               |        |        |        |     | NLGAD-SNP | 0.9002 | 0.8980 | 0.8979 | 0.8190 | 0.8176 | 0.8551 |
| ---------------- | --- | ------------- | ------ | ------ | ------ | --- | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| NLGAD-AAS 0.9272 |     | 0.9227 0.9358 | 0.8397 | 0.8099 | 0.8611 |     |           |        |        |        |        |        |        |
NLGAD-OLS 0.9031 0.8995 0.8974 0.7793 0.7853 0.7836 NLGAD 0.9286 0.9320 0.9457 0.8524 0.8435 0.8655
| NLGAD 0.9286 | 0.9320 | 0.9457 | 0.8524 | 0.8435 | 0.8655 |     |     |                     |     |     |     |     |     |
| ------------ | ------ | ------ | ------ | ------ | ------ | --- | --- | ------------------- | --- | --- | --- | --- | --- |
|              |        |        |        |        |        |     | 5.4 | SensibilityAnalysis |     |     |     |     |     |
Percentage𝐾
|     |     |     |     |     |     |     | 5.4.1 |     | ofNodesinNormalitySelection. |     |     |     | Figure4il- |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | ---------------------------- | --- | --- | --- | ---------- |
outperformstheothersonalldatasets.Theresultsshowthatthe lustratestheinfluenceofthepercentage𝐾 ofnodesinnormality
normalitylearning-basedschemeiseffectiveforGAD. selectionondetectionperformancewhen𝐾 variesfrom0.2to1.0
7508

MM’23,October29–November3,2023,Ottawa,ON,Canada. JingcanDuanetal.
with step 0.2.𝐾 = 1.0 means that the model does not perform overfittedonthesedatasetswhen𝑇 issettoolarger.Inpractice,
𝑟
|                                                       |     |                  |                   |     | weset𝑇 =500onCora,UAI2010,andCiteSeer,𝑇 | =600onDBLP, |
| ----------------------------------------------------- | --- | ---------------- | ----------------- | --- | --------------------------------------- | ----------- |
| normalityselection.Weobservethattheperformanceshowsan |     |                  |                   |     | 𝑟                                       | 𝑟           |
| increasingtrendas𝐾                                    |     | increasesbefore𝐾 | 1.0.Thisshowsthat |     |                                         |             |
|                                                       |     |                  | =                 |     | Citation,andACM.                        |             |
normalitylearningneedsasufficientnumberofnormalnodesas
theinput.Themodelcannotlearnthepatternofnormalitywell
bythedeficiencyofnormalnodes.Inthemeantime,thenormality
learning-basedschemecanfurtherimprovedetectionperformance.
Wefix𝐾
to0.8onalldatasets.
|     |     |     |     |     | (a)NormalitySelectionSteps𝑇𝑠 (b)NormalityLearningEpochs𝑇𝑟 |     |
| --- | --- | --- | --- | --- | --------------------------------------------------------- | --- |
Figure4:Sensibilityanalysisofpercentstrategy𝐾w.r.t.AUC. Figure6:Sensibilityanalysesofnormalityselectionsteps𝑇
𝑠
andnormalitylearningepochs𝑇
𝑟 w.r.t.AUC.
| And DBLP, | Citation, | and ACM | are more sensitive | to𝐾 than |     |     |
| --------- | --------- | ------- | ------------------ | -------- | --- | --- |
LossBalanceParameter𝛼. Wediscussthevitalbalancepa-
| Cora,UAI2010,andCiteSeer.Weconductadditionalexperiments |     |     |     |     | 5.4.4 |     |
| ------------------------------------------------------- | --- | --- | --- | --- | ----- | --- |
rameter𝛼
tofurtherexplorethereasonsforthisphenomenon.Werandomly inthelossfunction.AsillustratedinFigure7,thein-
fluenceof𝛼
select𝑅percentnormalnodesasthetraininginputofthemodel ondetectionperformanceshowsafirstupwardand
(withoutnormalityselectionandlearning)when𝑅variesfrom0.2 thendownwardtrendonalldatasets.Themulti-scalecontrastive
strategyeffectivelydigsdifferentanomalousinformation,which
to1.0.AsshowninFigure5,theperformancesonDBLP,Citation,
|     |     |     |     |     | contributestotheGADtask.Inpractice,weset𝛼 | to0.6,0.6,0.9, |
| --- | --- | --- | --- | --- | ----------------------------------------- | -------------- |
andACMaregreatlyinfluencedby𝑅,whichissimilartotheimpact
of𝐾.Thisindicatesthatthedifficultytolearnthepatternofnormal 0.7,0.7,and0.7onCora,UAI2010,CiteSeer,DBLP,Citation,and
ACM,respectively.
nodesisdifferentondifferentdatasets.Andsuchthreedatasets
needmorereliablenormalnodesthantheotherdatasetsinthe
normalitylearningphase.
Figure5:Percentage𝑅ofnormalnodesw.r.t.AUC.
|                      |     |                       |               |     | Figure7:Lossbalanceparameter𝛼 w.r.t.AUCvalues. |     |
| -------------------- | --- | --------------------- | ------------- | --- | ---------------------------------------------- | --- |
| 5.4.2 NumberofSteps𝑇 |     | inNormalitySelection. | Thenweexplore |     |                                                |     |
𝑠
| theeffectof𝑇 | ontheperformance.Figure6ashows𝑇 |     |     |           |              |     |
| ------------ | ------------------------------- | --- | --- | --------- | ------------ | --- |
|              | 𝑠                               |     |     | 𝑠 hasdif- | 6 CONCLUSION |     |
ferentimpactsondifferentdatasets.ThecurveofCitationshows
Inthispaper,weexplainwhatnormalitylearningisinGADandhow
anupwardandthendownwardtrend.CiteSeersuffersfromlittle
itcanhelptodetectanomaliesingraph.Then,wedeviseanormality
| influenceof𝑇 | 𝑠.Theothersrisefirst,thenremainessentiallyfixed. |     |     |     |     |     |
| ------------ | ------------------------------------------------ | --- | --- | --- | --- | --- |
learning-basedframework,NLGAD.Extensiveexperimentsonsix
Theseindicatethattoomanystepsinnormalityselectioncannot
benchmarkdatasetsconfirmthatNLGADoutperformsthestate-
effectivelyimprovethedetectionabilitybutbringuselesscalcu- of-the-artapproaches.Inthefuture,wewillfurtherexplorethe
lations.Tobalanceperformanceandcomplexity,weset𝑇
𝑠 =200
strategyofnormalityselectionthatcanobtainsufficientandreliable
| onCora,UAI2010,andCiteSeer,𝑇 |     | 𝑠   | =300onDBLP,Citation,and |     |     |     |
| ---------------------------- | --- | --- | ----------------------- | --- | --- | --- |
normalnodes.
ACM.
| NumberofEpochs𝑇 |     |                        |               |     | ACKNOWLEDGMENTS |     |
| --------------- | --- | ---------------------- | ------------- | --- | --------------- | --- |
| 5.4.3           |     | 𝑟 inNormalityLearning. | Figure6bshows |     |                 |     |
theimpactof𝑇
𝑟 onthedetectionperformance.ThecurvesonCora ThisworkwassupportedbytheNationalKeyR&DProgramof
and ACM are less influenced by𝑇 𝑟. And most curves rise first China (project no. 2020AAA0107100) and the National Natural
andthenfall.Itmanifeststhatthemodelmaysufferfrombeing ScienceFoundationofChina(projectno.62325604).
7509

NormalityLearning-basedGraphAnomalyDetectionviaMulti-ScaleContrastiveLearning MM’23,October29–November3,2023,Ottawa,ON,Canada.
REFERENCES
(2023).
[1] DeyuBo,XiaoWang,ChuanShi,andHuaweiShen.2021.Beyondlow-frequency [23] YueLiu,XihongYang,SihangZhou,XinwangLiu,ZhenWang,KeLiang,Wenx-
informationingraphconvolutionalnetworks.InProceedingsoftheAAAIConfer- uanTu,LiangLi,JingcanDuan,andCancanChen.2023.HardSampleAware
enceonArtificialIntelligence,Vol.35.3950–3957. NetworkforContrastiveDeepGraphClustering.InProc.ofAAAI.
[2] MarkusMBreunig,Hans-PeterKriegel,RaymondTNg,andJörgSander.2000. [24] XuexiongLuo,JiaWu,AminBeheshti,JianYang,XiankunZhang,YuanWang,
LOF:identifyingdensity-basedlocaloutliers.InProceedingsofthe2000ACM andShanXue.2022.ComGA:Community-AwareAttributedGraphAnomaly
SIGMODinternationalconferenceonManagementofdata.93–104. Detection.InProceedingsoftheFifteenthACMInternationalConferenceonWeb
SearchandDataMining.657–665.
[3] ZiweiChai,SiqiYou,YangYang,ShiliangPu,JiarongXu,HaoyangCai,and
[25] XiaoxiaoMa,JiaWu,ShanXue,JianYang,ChuanZhou,QuanZSheng,HuiXiong,
WeihaoJiang.[n.d.].CanAbnormalitybeDetectedbyGraphNeuralNetworks?
andLemanAkoglu.2021.Acomprehensivesurveyongraphanomalydetection
([n.d.]).
withdeeplearning.IEEETransactionsonKnowledgeandDataEngineering(2021).
[4] KaizeDing,JundongLi,RohitBhanushali,andHuanLiu.2019.Deepanomaly
[26] SungwonPark,SungwonHan,SundongKim,DanuKim,SungkyuPark,Se-
detectiononattributednetworks.InProceedingsofthe2019SIAMInternational
unghoonHong,andMeeyoungCha.2021.Improvingunsupervisedimagecluster-
ConferenceonDataMining.SIAM,594–602.
ingwithrobustlearning.InProceedingsoftheIEEE/CVFConferenceonComputer
[5] JingcanDuan,SiweiWang,XinwangLiu,HaifangZhou,JingtaoHu,andHu
VisionandPatternRecognition.12278–12287.
Jin.2022. GADMSL:GraphAnomalyDetectiononAttributedNetworksvia
[27] ZhihaoPeng,HuiLiu,YuhengJia,andJunhuiHou.2021.Attention-drivengraph
Multi-scaleSubstructureLearning.arXivpreprintarXiv:2211.15255(2022).
clusteringnetwork.InProceedingsofthe29thACMinternationalconferenceon
[6] JingcanDuan,SiweiWang,PeiZhang,EnZhu,JingtaoHu,HuJin,YueLiu,
multimedia.935–943.
andZhibinDong.2023. Graphanomalydetectionviamulti-scalecontrastive
[28] ZhenPeng,MinnanLuo,JundongLi,HuanLiu,andQinghuaZheng.2018.
learningnetworkswithaugmentedview.InProceedingsoftheAAAIConference
ANOMALOUS:AJointModelingApproachforAnomalyDetectiononAttributed
onArtificialIntelligence,Vol.37.7459–7467.
Networks..InIJCAI.3513–3519.
[7] DivamGupta,RamachandranRamjee,NipunKwatra,andMuthianSivathanu.
[29] BryanPerozzi,RamiAl-Rfou,andStevenSkiena.2014.Deepwalk:Onlinelearning
2019.Unsupervisedclusteringusingpseudo-semi-supervisedlearning.InInter-
ofsocialrepresentations.InProceedingsofthe20thACMSIGKDDinternational
nationalConferenceonLearningRepresentations.
conferenceonKnowledgediscoveryanddatamining.701–710.
[8] ChangheeHan,LeonardoRundo,KoheiMurao,TomoyukiNoguchi,YukiShima-
[30] SanjeevRao,AnilKumarVerma,andTarunpreetBhatia.2021.Areviewonsocial
hara,ZoltánÁdámMilacski,SaoriKoshino,EvisSala,HidekiNakayama,and
spamdetection:Challenges,openissues,andfuturedirections.ExpertSystems
Shin’ichiSatoh.2021.MADGAN:UnsupervisedmedicalanomalydetectionGAN
withApplications186(2021),115742.
usingmultipleadjacentbrainMRIslicereconstruction.BMCbioinformatics22,2
[31] PrithvirajSen,GalileoNamata,MustafaBilgic,LiseGetoor,BrianGalligher,and
(2021),1–20.
TinaEliassi-Rad.2008.Collectiveclassificationinnetworkdata.AImagazine29,
[9] NicholasAHeard,DavidJWeston,KiriakiPlatanioti,andDavidJHand.2010.
3(2008),93–93.
Bayesiananomalydetectionmethodsforsocialnetworks.TheAnnalsofApplied
[32] JianhengTang,JiajinLi,ZiqiGao,andJiaLi.2022. RethinkingGraphNeural
Statistics4,2(2010),645–662.
NetworksforAnomalyDetection.arXivpreprintarXiv:2205.15508(2022).
[10] TianjinHuang,YulongPei,VladoMenkovski,andMykolaPechenizkiy.2021.
[33] HanghangTong,ChristosFaloutsos,andJia-YuPan.2006. Fastrandomwalk
Hop-countbasedself-supervisedanomalydetectiononattributednetworks.
withrestartanditsapplications.InSixthinternationalconferenceondatamining
arXivpreprintarXiv:2104.07917(2021).
(ICDM’06).IEEE,613–622.
[11] MingJin,YixinLiu,YuZheng,LianhuaChi,Yuan-FangLi,andShiruiPan.2021.
[34] MinWang,HaoYang,andQingCheng.2022.GCL:GraphCalibrationLossfor
ANEMONE:GraphAnomalyDetectionwithMulti-ScaleContrastiveLearning.In
TrustworthyGraphNeuralNetwork.InProceedingsofthe30thACMInternational
Proceedingsofthe30thACMInternationalConferenceonInformation&Knowledge
ConferenceonMultimedia.988–996.
Management.3122–3126.
[35] WenjunWang,XiaoLiu,PengfeiJiao,XueChen,andDiJin.2018. Aunified
[12] ThomasNKipfandMaxWelling.2016.Semi-supervisedclassificationwithgraph
weaklysupervisedframeworkforcommunitydetectionandsemanticmatching.
convolutionalnetworks.arXivpreprintarXiv:1609.02907(2016).
InPacific-AsiaConferenceonKnowledgeDiscoveryandDataMining.Springer,
[13] AtsutoshiKumagai,TomoharuIwata,andYasuhiroFujiwara.2021. Semi-
218–230.
supervisedanomalydetectiononattributedgraphs.In2021InternationalJoint
[36] XuhongWang,BaihongJin,YingDu,PingCui,YingshuiTan,andYupuYang.
ConferenceonNeuralNetworks(IJCNN).IEEE,1–8.
2021. One-classgraphneuralnetworksforanomalydetectioninattributed
[14] JiongqianLiang,PeterJacobs,JiankaiSun,andSrinivasanParthasarathy.2018.
networks.Neuralcomputingandapplications33,18(2021),12073–12085.
Semi-supervisedembeddinginattributednetworkswithoutliers.InProceedings
[37] ZonghanWu,ShiruiPan,FengwenChen,GuodongLong,ChengqiZhang,and
ofthe2018SIAMinternationalconferenceondatamining.SIAM,153–161.
SYuPhilip.2020. Acomprehensivesurveyongraphneuralnetworks. IEEE
[15] ChangshuLiu,LiangjianWen,ZhaoKang,GuangchunLuo,andLingTian.2021.
transactionsonneuralnetworksandlearningsystems32,1(2020),4–24.
Self-supervisedconsensusrepresentationlearningforattributedgraph.InPro-
[38] RoseYu,HuidaQiu,ZhenWen,ChingYungLin,andYanLiu.2016. Asurvey
ceedingsofthe29thACMInternationalConferenceonMultimedia.2654–2662.
onsocialmediaanomalydetection.ACMSIGKDDExplorationsNewsletter18,1
[16] KayLiu,YingtongDou,YueZhao,XueyingDing,XiyangHu,RuitongZhang,
(2016),1–14.
KaizeDing,CanyuChen,HaoPeng,KaiShu,etal.2022.Bond:Benchmarking
[39] XuYuan,NaZhou,ShuoYu,HuafeiHuang,ZhikuiChen,andFengXia.2021.
unsupervisedoutliernodedetectiononstaticattributedgraphs. Advancesin
Higher-orderStructureBasedAnomalyDetectiononAttributedNetworks.In
NeuralInformationProcessingSystems35(2022),27021–27035.
2021IEEEInternationalConferenceonBigData(BigData).IEEE,2691–2700.
[17] YixinLiu,MingJin,ShiruiPan,ChuanZhou,YuZheng,FengXia,andPhilipYu.
[40] KunZhanandChaoxiNiu.2021. Mutualteachingforgraphconvolutional
2022.Graphself-supervisedlearning:Asurvey.IEEETransactionsonKnowledge
networks.FutureGenerationComputerSystems115(2021),837–843.
andDataEngineering(2022).
[41] JiaqiangZhang,SenzhangWang,andSongcanChen.2022. Reconstruction
[18] YixinLiu,ZhaoLi,ShiruiPan,ChenGong,ChuanZhou,andGeorgeKarypis.
EnhancedMulti-ViewContrastiveLearningforAnomalyDetectiononAttributed
2021.Anomalydetectiononattributednetworksviacontrastiveself-supervised
Networks.arXivpreprintarXiv:2205.04816(2022).
learning.IEEEtransactionsonneuralnetworksandlearningsystems(2021).
[42] YuZheng,MingJin,YixinLiu,LianhuaChi,KhoaTPhan,andYi-PingPhoebe
[19] YueLiu,KeLiang,JunXia,SihangZhou,XihongYang,,XinwangLiu,andZ.Stan
Chen.2021. GenerativeandContrastiveSelf-SupervisedLearningforGraph
Li.2023.Dink-Net:NeuralClusteringonLargeGraphs.InProc.ofICML.
AnomalyDetection.IEEETransactionsonKnowledgeandDataEngineering(2021).
[20] YueLiu,WenxuanTu,SihangZhou,XinwangLiu,LinxuanSong,XihongYang,
[43] ShuangZhou,QiaoyuTan,ZhimingXu,XiaoHuang,andFu-laiChung.2021.
andEnZhu.2022.DeepGraphClusteringviaDualCorrelationReduction.In
SubtractiveAggregationforAttributedNetworkAnomalyDetection.InPro-
ProceedingsoftheAAAIConferenceonArtificialIntelligence,Vol.36.7603–7611.
ceedingsofthe30thACMInternationalConferenceonInformation&Knowledge
[21] YueLiu,JunXia,SihangZhou,SiweiWang,XifengGuo,XihongYang,KeLiang,
Management.3672–3676.
WenxuanTu,Z.StanLi,andXinwangLiu.2022.ASurveyofDeepGraphClus-
[44] XiaokangZhou,WeiLiang,WeiminLi,KeYan,ShoheiShimizu,IKevin,andKai
tering:Taxonomy,Challenge,andApplication.arXivpreprintarXiv:2211.12875
Wang.2021.Hierarchicaladversarialattacksagainstgraphneuralnetworkbased
(2022).
IoTnetworkintrusiondetectionsystem.IEEEInternetofThingsJournal(2021).
[22] YueLiu,XihongYang,SihangZhou,andXinwangLiu.2023.Simplecontrastive
graphclustering.IEEETransactionsonNeuralNetworksandLearningSystems
7510

MM’23,October29–November3,2023,Ottawa,ON,Canada. JingcanDuanetal.
| A ALGORITHM B | COMPLEXITYCOMPARISON |     |     |
| ------------- | -------------------- | --- | --- |
TheoverallproceduresofNLGADareshowninAlgorithm1. WeconducttimecomplexityanalysisoftheSOTAGADmodels.
| 𝑇   | 𝑠+𝑇 𝑟 isthetotaltrainingepochsofNLGAD.And𝑇 |     | isthenumber |
| --- | ------------------------------------------ | --- | ----------- |
Algorithm1TheproposedNLGAD. ofepochsfortheothermodels.𝑟 isthenumberofroundsinthe
Input:AnundirectedgraphG=(V,E);Numberofnormalityselection inferencephase.Theothersymboldefinitionsareconsistentwith
steps𝑇𝑠;Numberofrefinetrainingepochs𝑇𝑟;Batchsize𝐵. Subsection4.4.Table6showsthatourmethodimprovesperfor-
Output:Anomalyscorefunction𝑓 (·). mancewithoutsignificantlyincreasingcomplexity.
1: Initializethemodelwithsubgraph-nodeandnode-nodecontrasts.
for𝑡 =1to𝑇𝑠do Table6:Comparisonoftimecomplexityfordifferentanom-
2:
Visdividedintobatcheswithsize𝐵byrandom.
| 3: alydetectionmethodsinGAD. |     |     |     |
| ---------------------------- | --- | --- | --- |
for𝑣𝑖 ∈𝐵do
4:
5: Ineachcontrast,estimatethesimilarityofthetargetnodeand
|     | Method | TimeComplexity |     |
| --- | ------ | -------------- | --- |
counterpartsinpositiveandnegativepairs.
6: Calculatetheanomalydegreeestimationsforeachnode. LOF[2] O(𝑛log𝑛)
O(cid:0)𝑛2𝑑(cid:1)
7: Selecthigh-confidentanomalydegreeestimationintothe ANOMALOUS[28]
normalitypool. DOMINANT[4] O(cid:0)(cid:0)𝐾𝑚𝑑+𝐾𝑛𝑑2(cid:1)𝑇(cid:1)
8: Calculatethejointlossofsubgraph-nodeandnode-node O(cid:0)𝑛(cid:0)𝑐𝛿+𝐾𝑞𝑑+𝐾𝑐𝑑2(cid:1)(𝑇 +𝑟)(cid:1)
CoLA[18]
| contrasts. |     | O(cid:0)𝑛(cid:0)𝑐𝛿+𝐾𝑞𝑑+𝐾𝑐𝑑2(cid:1)(𝑇 | +𝑟)(cid:1) |
| ---------- | --- | ------------------------------------ | ---------- |
ANEMONE[11]
9 : Ba c k propagationandupdatetrainableparameters. O(cid:0)𝑛(cid:0)𝑐𝛿+𝐾𝑞𝑑+𝐾𝑐𝑑2(cid:1)(𝑇 +𝑟)(cid:1)
SL-GAD[42]
10 : end f o r
| if𝑡 =𝑇𝑠then | HCM[10] | O(𝑚+(𝑚+𝑛)log𝑛+𝐾𝑚𝑑𝑇) |     |
| ----------- | ------- | ------------------- | --- |
11:
|     | Sub-CR[41] | O(cid:0)𝑛3+𝑛(cid:0)𝑐𝛿+𝐾𝑞𝑑+𝐾𝑐𝑑2(cid:1)(𝑇 | +𝑟)(cid:1) |
| --- | ---------- | --------------------------------------- | ---------- |
12: Averagethemeananomalydegreeestimationsforeachnode.
Assign𝐾percentnodeswithnormalpseudolabels.
|     |                 | O(cid:0)𝑛(cid:0)𝑐𝛿+𝐾𝑞𝑑+𝐾𝑐𝑑2(cid:1)(𝑇 | +𝑇 +𝑟)(cid:1) |
| --- | --------------- | ------------------------------------ | ------------- |
|     | NLGAD(Proposed) |                                      | 𝑠 𝑟           |
13: endif
14: endfor
for𝑡 =1to𝑇𝑟
15: do
16: Refinethemodelwiththeinputofselectednormalnodes.
17: endfor
18: Bymultiplerounddetections,calculatethefinalanomalyscorefor
eachnode.
7511