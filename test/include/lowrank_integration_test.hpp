// cpprate: Variable Selection in Black Box Methods with RelATive cEntrality (RATE) Measures
// https://github.com/tmaklin/cpprate
// Copyright (c) 2023 Tommi Mäklin (tommi@maklin.fi)
//
// BSD-3-Clause license
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     (1) Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//     (2) Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in
//     the documentation and/or other materials provided with the
//     distribution.
//
//     (3)The name of the author may not be used to
//     endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Test integration with low-rank factorization
#ifndef CPPRATE_LOWRANK_INTEGRATION_TEST_HPP
#define CPPRATE_LOWRANK_INTEGRATION_TEST_HPP

#include <cstddef>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/Dense>

#include "gtest/gtest.h"

#include "CppRateRes.hpp"

class LowrankIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
	// Parameters
	this->n_design_dim = 20;
	this->n_f_draws = 100;
	this->n_obs = 10;
	this->rank_r = 10;
	this->prop_var = 1.1;

	// Input data
	this->design_matrix = vec_to_sparse_matrix<double, bool>(std::vector<bool>(
			      { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
				0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
				1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
				0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,
				0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
				0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
				0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,
				0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,
				0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,
				0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0
			      }), this->n_obs, this->n_design_dim);
	this->f_draws = vec_to_dense_matrix(std::vector<double>({
		          -6.73547458213277,-0.300263985541422,-6.81470954432113,6.94318986278777,-5.44896509510378,-2.5366205703406,-1.24844768934632,8.77513237360593,-3.62467053670517,-0.489010242924885,
			  -6.81121711079486,0.998915389057232,-4.38573784402282,8.82734941393486,-4.43614826756895,-4.01327768662203,-1.41498699340357,9.53057970875737,-1.77382997040734,-1.1475687354857,
			  -11.262934255259,-0.810164429016605,-3.40483312131745,8.10505900545574,-4.75122561242841,-3.43652095823485,-0.580339883818108,6.73559624918973,-2.7181468122588,-1.68706486889551,
			  -7.18460544796125,0.411101506842434,-3.69836773990994,8.44999767869517,-6.19478559245962,-3.43603443271895,-0.80173053102948,9.35368080747247,-1.9153097506837,0.224677487797558,
			  -7.42536987603256,1.50389646565577,-3.55626309786152,7.80104807516072,-4.45218140243174,-3.19603759692659,-0.355626346493957,8.28956802601565,-2.02399614991911,-2.68261749993927,
			  -7.71118634134793,0.676919555436972,-3.89185759172763,7.44322950107747,-4.68826346697533,-2.63157188676209,-0.335024789210033,7.65216066637632,-3.49016988286547,-1.54741692455149,
			  -8.03995476324227,-0.268579997537278,-3.70990399533667,10.0253018445979,-6.67720624861776,-6.04878493365199,-0.991310530996322,8.65406427865356,-2.42517214388806,-1.96657029922516,
			  -8.23888397771089,0.509926787381964,-2.94647721808348,8.3120688118503,-4.67433266414377,-3.59247696278337,0.173545526612154,8.8830859096883,-2.72890118201985,-0.228647393286753,
			  -7.83213338836791,-0.810897719749218,-3.73544267014525,8.27002398114624,-6.2203821898954,-2.72601080527329,0.277786159275511,9.55149054172447,-2.15873870011567,-1.71371025713246,
			  -8.60007892940413,0.107761949222992,-4.91601675070345,9.17454973368119,-5.56926066279546,-2.4644103733495,0.792941196671334,8.51980594729676,-3.03093096496161,1.21317587013578,
			  -8.4238749851835,2.04816837405725,-1.62175571913268,9.29855616335592,-6.1331446240771,-3.63499306746155,2.05511799163502,9.13730918065625,-2.60552442327385,-2.69192849624559,
			  -9.5995334540179,-0.911616398933958,-3.49111691393217,9.37258904160051,-4.84968246560744,-2.98315286300446,-0.956135401022449,10.6037454079519,-4.27966166010779,-0.66248580959639,
			  -7.82477155246524,-0.516789645698781,-3.40979146620616,7.47570753449639,-4.27201222477138,-5.07960651908617,-0.329271769987626,8.56847019152688,-1.19806247939623,-0.888333707578184,
			  -7.97665937994288,-0.193622068496406,-4.05389381556064,8.6165566353229,-5.39891630461526,-3.45529142440548,1.17755212092688,8.74231903264143,-3.01054033195702,-1.74242496571166,
			  -6.94361687222457,-1.86513119976471,-3.01225132343187,10.3431054399987,-4.42086304499882,-2.40902518114314,-1.29646385668623,9.5184991972589,-3.853960927401,-2.58300483167032,
			  -9.42123742882223,-1.42002062562124,-4.41068620009801,8.7760382810633,-5.15091829726779,-3.88694439065688,-1.89967056731167,9.22907532234697,-4.47530828773633,-2.23491688233249,
			  -9.39803628884745,-1.33983291205289,-3.85790176601147,9.97917628697858,-5.74514367302002,-6.03308279864898,-0.386266692266024,10.1615982281319,-3.49334819414277,-0.656713531084458,
			  -10.1661909726279,0.625155678746472,-4.48893389604917,7.27293776935912,-4.44311132604895,-7.06050385623908,0.138892151906723,7.78346865694316,-2.17774392128173,-1.67361064522978,
			  -9.80405501909254,0.82869819635528,-2.66446893823018,8.3865992262248,-3.41544315556409,-1.22708352091626,-1.24398684898768,10.5280012969087,-4.60286031133683,-2.44193509662128,
			  -8.27913825574939,0.587220848573846,-2.0854862732303,7.76086578411509,-4.39829575734499,-4.15168799785145,2.35646931721459,8.64939871612326,-3.674755220729,-2.18974533942523,
			  -8.12333511690693,0.283291812796853,-4.68959334402997,7.16121848290397,-4.36378806475545,-3.95036231080033,-0.983633576237723,9.42747869511366,-3.36859333574084,-2.2126662032603,
			  -7.47535732159545,-0.642910174752912,-2.62890012313538,9.0639830967738,-6.36056302870526,-4.99376281548742,0.669432566849132,8.47947193868778,-1.59795642533064,-1.89221631001722,
			  -7.96460878586186,0.435178445131722,-3.2834194327937,6.64096681091012,-5.29524342703971,-2.53761579806467,-0.672015089648073,7.99664335431368,-2.81071136892509,-1.19359678509643,
			  -8.01064869531377,0.856087863673504,-4.25879204993957,9.27985549084926,-5.51761912298599,-3.87488913145133,-0.0415549686119719,9.56540700374841,-2.54131261172151,-1.37488830756328,
			  -9.62414383613782,0.686572331945818,-4.15857154135951,8.13309428250268,-5.72854023958486,-3.19567821056462,0.349499103566149,5.9335755946641,-3.23854145228238,-2.87885844479073,
			  -9.64951454568365,0.492087997054664,-3.509863604877,6.64899615149859,-6.7751822843734,-6.59196209441196,1.20573497606954,7.90521700672589,-1.75828255860133,-0.109976480149243,
			  -8.54284609587024,-1.93188551351571,-3.90944805444785,7.95800442568459,-5.08109041463254,-4.24341479899939,-1.20534652663729,7.60443899652531,-3.31910406309969,-0.968843241897904,
			  -8.35963833198398,1.84927359392299,-5.32503923230809,8.68996184695166,-5.04132555868765,-6.16255664335793,-0.0432116713285828,9.01324546135903,-1.71399933761104,-2.52365268064629,
			  -8.16528409508592,-0.478618708478723,-2.73166446812264,8.61400439573847,-4.42894107666852,-4.54380589821518,0.900726361118607,7.71146128837565,-1.56567443847898,0.29319600732247,
			  -6.86259638471967,1.07601281550838,-2.51943936580931,7.58537495952486,-5.11809973663673,-5.14090196075309,-1.67754205605507,7.66795407523482,-1.72263303718251,-0.825040105553593,
			  -7.56958566408279,-1.55801147482336,-4.78294670992509,8.16739562439798,-4.66959051738059,-3.09209604884898,-0.642347429770996,8.46478671326643,-3.66833261466607,-2.13157506283179,
			  -7.07952533316105,0.383312043774486,-4.55419077780727,8.93424382513575,-5.19753783383349,-4.72857510412071,-1.23745521491091,7.91620253678152,-2.60165551832967,-2.15475298393857,
			  -7.44167519045584,-0.612062425971596,-3.69825323456155,10.0352979416387,-4.75244324261732,-4.60723451752368,-1.60853770550557,8.92040979381036,-2.00396948975288,-2.1344065784154,
			  -8.42641888023765,1.56870872688097,-3.47602353448392,8.42059311335488,-4.7431367241909,-3.65682092881721,-0.304603353766509,8.83362552471935,-1.63756751094161,0.541287641852998,
			  -8.85833067719863,-2.02689931789878,-4.61428662430993,9.71298322928006,-4.12050013720944,-2.81907806601561,0.296179959558497,9.11816251134571,-2.9957957807499,-1.97895260502359,
			  -7.32028364098823,1.00496004240818,-2.93389098397772,9.87100489071915,-6.2559999642646,-4.81512307846087,1.95760832842923,6.58940312982806,-2.08628185249003,-1.12962373508743,
			  -8.89945004805696,-1.56512988738363,-2.11949704630227,10.6149324419629,-6.40985149326872,-4.42866723803191,-1.89270332640354,9.06447116621196,-2.54858778169807,-0.754922497432813,
			  -6.70098784279356,-0.327390830801168,-4.54563285937597,7.37829270106081,-5.87175730412969,-3.27568043862357,0.285587234757797,9.70637286163723,-2.26190023146392,-2.18438385528561,
			  -8.25077189749296,-0.46537420356065,-5.63431269310118,6.7007305777919,-4.99634920997263,-4.19653813425869,-0.0884493711996349,8.69986842343325,-1.97911787936973,-1.63747012312203,
			  -5.59823117890053,-0.934777823602583,-2.86574830503972,7.37125677925353,-2.35306670530865,-3.92287621510898,0.470257687673675,9.60061133387152,-1.56167027309081,-0.451502036075558,
			  -8.96514530337018,-0.774380681689025,-4.09290401684309,7.59830443216785,-7.48689940176302,-3.82596400157762,-0.373932731314759,10.0522437543681,-1.08071243237054,-2.21711074983377,
			  -7.24383034920484,0.457737964377537,-3.15777512780505,8.14033587417589,-6.18125126366166,-2.4833854806486,-0.886695939875846,8.96241327100089,-3.46779053718137,0.109858574624932,
			  -5.96232581955443,0.484267367670058,-3.19081120982071,7.64997751030948,-4.82174825330661,-4.49438442748874,-0.899659398981922,9.85584381740381,-5.43821310250935,-1.02700287072494,
			  -6.66764085308926,0.568251329452562,-3.57713043708985,11.3502658761489,-4.34819639650459,-5.40118070172923,-0.632421345920192,9.33913646440156,-1.73291132453883,-0.110268297070596,
			  -7.36742710849262,-0.772253891566379,-4.40485345517429,8.07835542132137,-6.6295119854865,-4.33884202668686,-0.429490036292798,9.90235176932215,-3.71289258564055,0.012013646911712,
			  -7.12745837223298,-0.424541231190692,-5.39659799779607,9.73168574533252,-6.6412444124488,-3.20036406369664,-0.938506063564527,6.7759849299817,-2.21407763579736,0.365242080276623,
			  -9.0308724476995,0.141637296923665,-3.78791318460528,9.01478812037977,-3.69582008873901,-4.50493717297693,-0.529591712247864,9.44674408559133,-1.35568195917569,-1.78432053756739,
			  -10.0137522558452,0.547324519004524,-3.26197528882737,6.46928134747538,-4.42754388815695,-4.68478576784196,0.168424624837695,8.83373695674929,-3.65977870270156,-1.71835427593158,
			  -6.32785735865175,-0.665588291977814,-3.44165697923693,6.36722328130987,-4.31679685798269,-4.50300657456381,0.214305211021733,10.2258347449813,-3.3173485994862,-0.5962310177045,
			  -10.0800571948969,0.440121928126289,-3.57018083320968,7.8168089044562,-6.09317488844121,-4.22352637146442,0.0686588170783416,9.91335526844985,-2.12642515470793,-0.652271445499048,
			  -5.25170066226124,0.64418327905306,-4.7991584269962,9.42885688250979,-4.37551937679465,-3.63829042116599,-1.42504812911678,9.07888470988358,-2.25730590597918,0.174174711396046,
			  -7.30521776974686,-0.0474758384828119,-5.92766181579151,6.65874600738482,-4.84815046636847,-4.80045959375144,-0.650149721283366,10.8067679683642,-3.46560675439642,-0.467943100903515,
			  -7.0082855945731,0.0786490994153827,-3.63445871253626,8.21923027490698,-5.29753466214166,-2.66691090028838,-2.16661601341159,7.68587237240697,-1.62770343975497,-2.03033276564432,
			  -5.82955411140246,0.328814580143923,-3.96456103218641,10.0814549452484,-6.22263790810133,-4.76979398489753,0.0594708758660472,8.49726730704965,-0.595563814636188,-1.84962051594304,
			  -8.86220621676383,-2.09005591245504,-3.89811830826603,8.81780176727217,-5.36796675311138,-2.40428918832165,-0.0960970126034026,11.2775362004959,-1.5202243357932,-0.175472765730101,
			  -7.50116175414039,-1.83635496757947,-3.14099011531955,7.81534687352204,-4.03406580886016,-3.29572953223845,0.308962358365492,8.99437210507069,-2.75059329867142,-1.20413203178382,
			  -7.8089208474507,-1.4630616744322,-4.54779929139304,7.81289316658046,-4.99245493877343,-5.16457323248979,1.48070657104382,9.30602288606873,-2.41195724908914,-0.747377564193131,
			  -8.15290404290716,2.39865897229638,-3.51964739056379,10.4020260473646,-4.45039165728873,-4.5929488957831,-1.62553614058629,9.75102667697881,-2.43510670296487,-1.24814824380887,
			  -7.61764907951161,-1.40877179609202,-4.3987502858666,7.99300670323774,-4.4851475722254,-2.54971463086518,-1.65737884290182,10.2842970579209,-3.345311989305,-1.46678506693482,
			  -6.9729960266329,-0.866171147611734,-3.66393926734356,7.51762887765254,-6.01703157022628,-4.42163581412321,-0.634937706934717,9.9007628409984,-2.40085231332794,-1.53685256229713,
			  -7.58979729034705,-0.411932589903625,-5.1289844521241,8.70536647893505,-5.76840433628518,-4.20683589311562,-0.756348301869077,8.85171929747348,-5.76721107668329,-0.627823955137088,
			  -8.91113511954649,-0.346530684209406,-3.82496590905199,7.69953882262067,-6.08671351991342,-4.0842752852991,-1.06497341527198,8.78080377029104,-2.32725604868839,-0.0848521899982522,
			  -8.2861611586094,-0.649807155750468,-4.25073758181415,7.06877020420774,-3.54348965109184,-2.57057212984276,0.0572698275381159,8.49837163641549,-2.97580220740161,-0.89267307740774,
			  -9.02139393428003,-0.234380149035932,-3.72267482439957,8.3038359212208,-5.74288982432295,-4.52955271546239,-1.50471061337777,9.20008541012083,-3.12812371109687,-0.758700139253588,
			  -6.73035899318426,0.603764068931288,-3.32017371422742,8.10296077503879,-4.42387728445852,-3.71048365088539,-1.35970822983911,8.2154013736644,-3.9922919314321,-2.29413928582887,
			  -7.92125337515567,-1.13639321954203,-2.22827566149023,8.69405990871262,-5.46391737491763,-4.24976197032967,1.52321550994131,10.1472014139064,-1.12847213162909,-2.20069053453918,
			  -8.65557703288185,0.00878321147419582,-5.99298186824395,9.23022780521305,-4.50726608435735,-4.99403623536499,0.900361291294057,8.35129566787828,-2.00938606294333,-1.8564926643872,
			  -8.01143023174295,0.172880464830976,-3.24244132303678,8.33035096835676,-5.73872705936749,-2.43536212472317,1.99337966997579,9.88798391469967,-3.60428539725936,-1.1513452446957,
			  -7.08053828538639,-1.38745359591809,-3.61401167976106,8.77783453532865,-3.4820248081304,-4.73589930009675,-0.513978899474451,7.45759957880426,-1.864904105301,-2.51741726301119,
			  -8.36839995408138,-1.05951732171848,-3.46990044334151,8.47157365551674,-4.5412479242409,-4.20793377786821,-0.568737441618642,10.00350753772,-2.18971236722735,-0.229813526004511,
			  -8.13980286102473,-0.612431442190555,-3.41821411207087,9.25275740620337,-4.61216512189627,-4.33841706649141,-0.861965619881899,6.94994101613835,-3.97199056679625,-1.21552007740638,
			  -6.67838075330651,-1.51129788803131,-3.20431333289674,8.71771290401048,-4.86865091731151,-4.52545309788984,0.733914781655638,8.17315726103543,-3.17394172056184,-1.85111627155778,
			  -8.76458460027659,1.08265385442885,-2.86013853140988,8.78987302698738,-5.20238130131828,-4.87027478836341,-0.914537530765042,8.52529094027972,-2.03901505757839,-1.33744157440247,
			  -7.93474747213012,2.39775593513979,-4.78939462752427,9.21843472806285,-5.13825572910859,-4.01215678688559,-0.320443504853683,9.39935470153349,-3.89317158524778,-1.05956014234145,
			  -7.81244321744293,-0.610334485159577,-4.65003045536218,8.68983533031733,-5.35601439789761,-5.02340978063261,-0.0982768428491813,10.3283829423435,-2.96236166577385,-1.72623093110891,
			  -6.8585214195656,-0.411004287782794,-4.43017966043943,8.29609651669644,-5.99484719200418,-4.62341298284563,1.99656892161552,8.66248019865916,-2.51597815049746,-1.44263275735853,
			  -6.75257321750977,-0.296626511019321,-5.1385326984773,8.78711392129352,-4.50697445962657,-3.34046274050122,-0.173451091244495,7.99337861510504,-1.53075936851245,-1.42846638429673,
			  -6.94198885812699,-0.798555595059907,-2.6757909372964,9.70633925068056,-4.15980926182616,-3.13934075514573,0.0974666091388215,7.2892681347252,-4.27574601662799,-2.21086386824889,
			  -7.76458778395803,-0.717757662880755,-3.86041849646296,9.53299093197364,-4.713404203369,-5.39517526828556,-0.308516460551183,9.11047033855408,-1.02250422618792,-1.09662195057177,
			  -8.37284701993591,-1.273391214806,-2.90693494166971,9.72741647520181,-5.12746799403193,-5.16338633492384,0.188581262080357,6.76629301770505,-3.58372248063905,-1.47255724944967,
			  -8.36872858540188,-0.184144136406493,-4.40902172564164,9.06164080967947,-5.55974074460273,-4.30986120915408,0.178119179559627,9.58006653357951,-1.22755204162142,-2.49026944482244,
			  -7.23622498951099,-0.96023277262944,-3.20173312370427,7.66198986237773,-4.63003104614801,-4.1920538214646,1.56933938086658,9.93490159625351,-2.00034327047101,0.245995918900533,
			  -7.89948884988887,1.20151282219269,-4.79132113176763,8.05415265609404,-4.48314450405031,-3.37954357321204,0.277820912516352,9.61400643975065,-3.44905752961165,-0.110180215147973,
			  -7.30693940024504,0.96405680747379,-4.32989246516653,8.18949435053579,-5.84647753169126,-3.74980357489348,-0.981357813926935,8.61232892529763,-1.89425698340093,-2.98745317589956,
			  -7.3616735699824,0.907756754527362,-4.32486367838385,8.89862864378946,-6.14240197326206,-2.73186257659278,0.484690558283143,9.37572925068601,-2.88312051790372,-1.27654576452665,
			  -9.5014234490802,-0.013636647210515,-1.953979573892,7.14309679511572,-3.99752621369654,-3.41815985414386,-1.99924890435916,7.03347848873572,-3.1029602244354,0.646393574910985,
			  -6.80142479701945,-0.0613511520954221,-2.34958517442236,7.91930482467772,-4.06240959087083,-5.1815011729939,-0.144756208332792,8.63410081786663,-1.22977931385868,-0.482566025756041,
			  -8.07341801874838,-0.327953180521493,-4.06146601428913,9.10348346717181,-4.74585334503515,-4.97010278462806,-0.486188031089316,9.07662177989339,-2.04996099766975,-1.51549785212585,
			  -8.32771625353111,0.247687216530314,-3.97174948576197,8.61110308664502,-5.97690228599318,-3.81569990930754,-1.31871725280948,6.85460975277952,-2.46615597453519,-1.03150156238744,
			  -8.80151324985168,1.57381416930428,-3.83958926226986,7.13710989488203,-4.67095316500724,-2.73877256101654,-0.893642021298777,9.13391056104506,-2.19888340632346,-0.751399431802071,
			  -6.59780293411598,-1.48143683500582,-3.08214630726115,8.40929135769838,-4.04320522902627,-5.8106939479995,0.490211480603684,8.27680584821406,-2.91513889124954,-0.490808419799687,
			  -6.64774712455701,-0.393582989049232,-3.10990418209433,8.08958605280724,-1.63260025105059,-4.6176179590342,-1.10677049019771,7.70634202965862,-4.05420387043102,-0.632772856036838,
			  -6.2634549454517,0.332746994124367,-5.64924927594095,9.09012670003025,-4.46637003540516,-4.37551854772024,-2.86158854104568,9.76677808254423,-1.47433672391651,2.90993514228545,
			  -7.73413760406898,-0.599106982412199,-3.99517754246961,8.23680656013317,-6.95679496056182,-4.18181097035545,-2.16171645389436,10.8127551052276,-2.97702630279568,-0.177521176415248,
			  -6.998151863674,-0.5037782516005,-2.36631521422807,6.60885654452087,-4.71516197819924,-2.44473178834911,0.178922521447033,10.1747799873246,-2.74073593699798,-0.694142538370626,
			  -7.32525222506493,0.16336673389982,-2.98387896861812,7.89921787898665,-4.20485402287793,-3.72155881952953,0.991945185767405,9.08384858329201,-3.20865808593071,-2.20056523897724,
			  -9.38540962370249,-1.61399303760486,-4.85801719649106,7.32367557223145,-3.07482109039817,-5.30662488840436,-0.0960134744816779,9.251406575123,-2.9996086741383,-1.2146487064817,
			  -9.33682255973776,0.703565385115437,-2.74282672205004,7.25337195483205,-5.0884862953915,-4.87045033355581,-1.4925912816456,8.93573427136695,-1.8015467505682,-1.54109088181852,
			  -9.90160160251689,0.71191842567529,-4.2672694164969,8.74520795962353,-3.39613878709751,-5.97160846028749,0.386374248155131,8.16164839076611,-2.6808737299427,-0.84249110100532,
			  -7.30610433687758,-1.11292889796795,-3.79580179012717,9.13856980899353,-4.78111293832235,-3.11861539671737,0.341072531319388,7.57812149238847,-3.83109100733496,-2.38914445286053 }),
		this->n_f_draws, this->n_obs);
    }

    void TearDown() override {
	this->n_design_dim = 0;
	this->n_f_draws = 0;
	this->n_obs = 0;
	this->rank_r = 0;
	this->prop_var = 0.0;

	this->design_matrix.resize(0, 0);
	this->f_draws.resize(0, 0);
    }

    // Test parameters
    size_t n_design_dim;
    size_t n_f_draws;
    size_t n_obs;
    size_t rank_r;
    double prop_var;

    // Expected values
    static std::vector<double> expected_lr_KLD;
    static std::vector<double> expected_lr_RATE;
    static double expected_lr_ESS;
    static double expected_lr_Delta;

    // Input values
    Eigen::SparseMatrix<double> design_matrix; // 10x`n_design_dims` matrix stored contiguously
    Eigen::MatrixXd f_draws; // 10x1 vector

};
std::vector<double> LowrankIntegrationTest::expected_lr_KLD = { 3813.4,438.409,165.135,165.135,1.9e-30,165.135,215.076,2359.94,165.135,511.741,5083.46,66.9146,3714.4,2359.94,66.9146,1.9e-30,8210.96,80.4416,165.135,1.9e-30 };
std::vector<double> LowrankIntegrationTest::expected_lr_RATE = { 0.137442,0.0157971,0.00595118,0.00595118,6.84767e-35,0.00595118,0.00774958,0.0850353,0.00595118,0.0184435,0.18318,0.00241162,0.133842,0.0850353,0.00241162,6.84767e-35,0.295997,0.00289927,0.00595118,6.84767e-35 };
double LowrankIntegrationTest::expected_lr_ESS = 50.3003;
double LowrankIntegrationTest::expected_lr_Delta = 0.98811682924338817;

#endif
