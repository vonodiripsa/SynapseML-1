"use strict";(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[4300],{3905:function(e,r,t){t.d(r,{Zo:function(){return c},kt:function(){return d}});var n=t(7294);function a(e,r,t){return r in e?Object.defineProperty(e,r,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[r]=t,e}function i(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}function s(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?i(Object(t),!0).forEach((function(r){a(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}function o(e,r){if(null==e)return{};var t,n,a=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(e,r);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var p=n.createContext({}),l=function(e){var r=n.useContext(p),t=r;return e&&(t="function"==typeof e?e(r):s(s({},r),e)),t},c=function(e){var r=l(e.components);return n.createElement(p.Provider,{value:r},e.children)},m={inlineCode:"code",wrapper:function(e){var r=e.children;return n.createElement(n.Fragment,{},r)}},u=n.forwardRef((function(e,r){var t=e.components,a=e.mdxType,i=e.originalType,p=e.parentName,c=o(e,["components","mdxType","originalType","parentName"]),u=l(t),d=a,g=u["".concat(p,".").concat(d)]||u[d]||m[d]||i;return t?n.createElement(g,s(s({ref:r},c),{},{components:t})):n.createElement(g,s({ref:r},c))}));function d(e,r){var t=arguments,a=r&&r.mdxType;if("string"==typeof e||a){var i=t.length,s=new Array(i);s[0]=u;var o={};for(var p in r)hasOwnProperty.call(r,p)&&(o[p]=r[p]);o.originalType=e,o.mdxType="string"==typeof e?e:a,s[1]=o;for(var l=2;l<i;l++)s[l]=t[l];return n.createElement.apply(null,s)}return n.createElement.apply(null,t)}u.displayName="MDXCreateElement"},3634:function(e,r,t){t.r(r),t.d(r,{frontMatter:function(){return o},contentTitle:function(){return p},metadata:function(){return l},toc:function(){return c},default:function(){return u}});var n=t(3117),a=t(102),i=(t(7294),t(3905)),s=["components"],o={title:"HyperParameterTuning - Fighting Breast Cancer",hide_title:!0,status:"stable"},p=void 0,l={unversionedId:"features/other/HyperParameterTuning - Fighting Breast Cancer",id:"version-0.9.5/features/other/HyperParameterTuning - Fighting Breast Cancer",title:"HyperParameterTuning - Fighting Breast Cancer",description:"HyperParameterTuning - Fighting Breast Cancer",source:"@site/versioned_docs/version-0.9.5/features/other/HyperParameterTuning - Fighting Breast Cancer.md",sourceDirName:"features/other",slug:"/features/other/HyperParameterTuning - Fighting Breast Cancer",permalink:"/SynapseML/docs/features/other/HyperParameterTuning - Fighting Breast Cancer",tags:[],version:"0.9.5",frontMatter:{title:"HyperParameterTuning - Fighting Breast Cancer",hide_title:!0,status:"stable"},sidebar:"docs",previous:{title:"DeepLearning - Transfer Learning",permalink:"/SynapseML/docs/features/other/DeepLearning - Transfer Learning"},next:{title:"TextAnalytics - Amazon Book Reviews with Word2Vec",permalink:"/SynapseML/docs/features/other/TextAnalytics - Amazon Book Reviews with Word2Vec"}},c=[{value:"HyperParameterTuning - Fighting Breast Cancer",id:"hyperparametertuning---fighting-breast-cancer",children:[],level:2}],m={toc:c};function u(e){var r=e.components,t=(0,a.Z)(e,s);return(0,i.kt)("wrapper",(0,n.Z)({},m,t,{components:r,mdxType:"MDXLayout"}),(0,i.kt)("h2",{id:"hyperparametertuning---fighting-breast-cancer"},"HyperParameterTuning - Fighting Breast Cancer"),(0,i.kt)("p",null,"We can do distributed randomized grid search hyperparameter tuning with SynapseML."),(0,i.kt)("p",null,"First, we import the packages"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"import pandas as pd\n\n")),(0,i.kt)("p",null,"Now let's read the data and split it to tuning and test sets:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'data = spark.read.parquet("wasbs://publicwasb@mmlspark.blob.core.windows.net/BreastCancer.parquet").cache()\ntune, test = data.randomSplit([0.80, 0.20])\ntune.limit(10).toPandas()\n')),(0,i.kt)("p",null,"Next, define the models that wil be tuned:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.automl import TuneHyperparameters\nfrom synapse.ml.train import TrainClassifier\nfrom pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier\nlogReg = LogisticRegression()\nrandForest = RandomForestClassifier()\ngbt = GBTClassifier()\nsmlmodels = [logReg, randForest, gbt]\nmmlmodels = [TrainClassifier(model=model, labelCol="Label") for model in smlmodels]\n')),(0,i.kt)("p",null,"We can specify the hyperparameters using the HyperparamBuilder.\nWe can add either DiscreteHyperParam or RangeHyperParam hyperparameters.\nTuneHyperparameters will randomly choose values from a uniform distribution."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"from synapse.ml.automl import *\n\nparamBuilder = \\\n  HyperparamBuilder() \\\n    .addHyperparam(logReg, logReg.regParam, RangeHyperParam(0.1, 0.3)) \\\n    .addHyperparam(randForest, randForest.numTrees, DiscreteHyperParam([5,10])) \\\n    .addHyperparam(randForest, randForest.maxDepth, DiscreteHyperParam([3,5])) \\\n    .addHyperparam(gbt, gbt.maxBins, RangeHyperParam(8,16)) \\\n    .addHyperparam(gbt, gbt.maxDepth, DiscreteHyperParam([3,5]))\nsearchSpace = paramBuilder.build()\n# The search space is a list of params to tuples of estimator and hyperparam\nprint(searchSpace)\nrandomSpace = RandomSpace(searchSpace)\n")),(0,i.kt)("p",null,"Next, run TuneHyperparameters to get the best model."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'bestModel = TuneHyperparameters(\n              evaluationMetric="accuracy", models=mmlmodels, numFolds=2,\n              numRuns=len(mmlmodels) * 2, parallelism=1,\n              paramSpace=randomSpace.space(), seed=0).fit(tune)\n')),(0,i.kt)("p",null,"We can view the best model's parameters and retrieve the underlying best model pipeline"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"print(bestModel.getBestModelInfo())\nprint(bestModel.getBestModel())\n")),(0,i.kt)("p",null,"We can score against the test set and view metrics."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"from synapse.ml.train import ComputeModelStatistics\nprediction = bestModel.transform(test)\nmetrics = ComputeModelStatistics().transform(prediction)\nmetrics.limit(10).toPandas()\n")))}u.isMDXComponent=!0}}]);