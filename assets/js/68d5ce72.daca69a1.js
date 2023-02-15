"use strict";(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[9835],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return u}});var l=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function r(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);t&&(l=l.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,l)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?r(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):r(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,l,o=function(e,t){if(null==e)return{};var n,l,o={},r=Object.keys(e);for(l=0;l<r.length;l++)n=r[l],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(l=0;l<r.length;l++)n=r[l],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var s=l.createContext({}),m=function(e){var t=l.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},p=function(e){var t=m(e.components);return l.createElement(s.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return l.createElement(l.Fragment,{},t)}},f=l.forwardRef((function(e,t){var n=e.components,o=e.mdxType,r=e.originalType,s=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),f=m(n),u=o,d=f["".concat(s,".").concat(u)]||f[u]||c[u]||r;return n?l.createElement(d,a(a({ref:t},p),{},{components:n})):l.createElement(d,a({ref:t},p))}));function u(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var r=n.length,a=new Array(r);a[0]=f;var i={};for(var s in t)hasOwnProperty.call(t,s)&&(i[s]=t[s]);i.originalType=e,i.mdxType="string"==typeof e?e:o,a[1]=i;for(var m=2;m<r;m++)a[m]=n[m];return l.createElement.apply(null,a)}return l.createElement.apply(null,n)}f.displayName="MDXCreateElement"},48840:function(e,t,n){n.r(t),n.d(t,{assets:function(){return p},contentTitle:function(){return s},default:function(){return u},frontMatter:function(){return i},metadata:function(){return m},toc:function(){return c}});var l=n(83117),o=n(80102),r=(n(67294),n(3905)),a=["components"],i={title:"Examples",description:"Examples using SynapseML with MLflow"},s=void 0,m={unversionedId:"mlflow/examples",id:"mlflow/examples",title:"Examples",description:"Examples using SynapseML with MLflow",source:"@site/docs/mlflow/examples.md",sourceDirName:"mlflow",slug:"/mlflow/examples",permalink:"/SynapseML/docs/next/mlflow/examples",draft:!1,tags:[],version:"current",frontMatter:{title:"Examples",description:"Examples using SynapseML with MLflow"},sidebar:"docs",previous:{title:"Mlflow Installation",permalink:"/SynapseML/docs/next/mlflow/installation"},next:{title:"SynapseML Autologging",permalink:"/SynapseML/docs/next/mlflow/autologging"}},p={},c=[{value:"Prerequisites",id:"prerequisites",level:2},{value:"API Reference",id:"api-reference",level:2},{value:"LightGBMClassificationModel",id:"lightgbmclassificationmodel",level:2},{value:"Cognitive Services",id:"cognitive-services",level:2}],f={toc:c};function u(e){var t=e.components,n=(0,o.Z)(e,a);return(0,r.kt)("wrapper",(0,l.Z)({},f,n,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("h2",{id:"prerequisites"},"Prerequisites"),(0,r.kt)("p",null,"If you're using Databricks, install mlflow with this command:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"# run this so that mlflow is installed on workers besides driver\n%pip install mlflow\n")),(0,r.kt)("p",null,"Install SynapseML based on the ",(0,r.kt)("a",{parentName:"p",href:"/SynapseML/docs/next/getting_started/installation"},"installation guidance"),"."),(0,r.kt)("h2",{id:"api-reference"},"API Reference"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.save_model"},"mlflow.spark.save_model")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.log_model"},"mlflow.spark.log_model")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.load_model"},"mlflow.spark.load_model")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric"},"mlflow.log_metric"))),(0,r.kt)("h2",{id:"lightgbmclassificationmodel"},"LightGBMClassificationModel"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'import mlflow\nfrom synapse.ml.featurize import Featurize\nfrom synapse.ml.lightgbm import *\nfrom synapse.ml.train import ComputeModelStatistics\n\nwith mlflow.start_run():\n\n    feature_columns = ["Number of times pregnant","Plasma glucose concentration a 2 hours in an oral glucose tolerance test",\n    "Diastolic blood pressure (mm Hg)","Triceps skin fold thickness (mm)","2-Hour serum insulin (mu U/ml)",\n    "Body mass index (weight in kg/(height in m)^2)","Diabetes pedigree function","Age (years)"]\n    df = spark.createDataFrame([\n        (0,131,66,40,0,34.3,0.196,22,1),\n        (7,194,68,28,0,35.9,0.745,41,1),\n        (3,139,54,0,0,25.6,0.402,22,1),\n        (6,134,70,23,130,35.4,0.542,29,1),\n        (9,124,70,33,402,35.4,0.282,34,0),\n        (0,93,100,39,72,43.4,1.021,35,0),\n        (4,110,76,20,100,28.4,0.118,27,0),\n        (2,127,58,24,275,27.7,1.6,25,0),\n        (0,104,64,37,64,33.6,0.51,22,1),\n        (2,120,54,0,0,26.8,0.455,27,0),\n        (7,178,84,0,0,39.9,0.331,41,1),\n        (2,88,58,26,16,28.4,0.766,22,0),\n        (1,91,64,24,0,29.2,0.192,21,0),\n        (10,101,76,48,180,32.9,0.171,63,0),\n        (5,73,60,0,0,26.8,0.268,27,0),\n        (3,158,70,30,328,35.5,0.344,35,1),\n        (2,105,75,0,0,23.3,0.56,53,0),\n        (12,84,72,31,0,29.7,0.297,46,1),\n        (9,119,80,35,0,29.0,0.263,29,1),\n        (6,93,50,30,64,28.7,0.356,23,0),\n        (1,126,60,0,0,30.1,0.349,47,1)\n    ], feature_columns+["labels"]).repartition(2)\n\n\n    featurize = (Featurize()\n    .setOutputCol("features")\n    .setInputCols(feature_columns)\n    .setOneHotEncodeCategoricals(True)\n    .setNumFeatures(4096))\n\n    df_trans = featurize.fit(df).transform(df)\n\n    lightgbm_classifier = (LightGBMClassifier()\n            .setFeaturesCol("features")\n            .setRawPredictionCol("rawPrediction")\n            .setDefaultListenPort(12402)\n            .setNumLeaves(5)\n            .setNumIterations(10)\n            .setObjective("binary")\n            .setLabelCol("labels")\n            .setLeafPredictionCol("leafPrediction")\n            .setFeaturesShapCol("featuresShap"))\n\n    lightgbm_model = lightgbm_classifier.fit(df_trans)\n\n    # Use mlflow.spark.save_model to save the model to your path\n    mlflow.spark.save_model(lightgbm_model, "lightgbm_model")\n    # Use mlflow.spark.log_model to log the model if you have a connected mlflow service\n    mlflow.spark.log_model(lightgbm_model, "lightgbm_model")\n\n    # Use mlflow.pyfunc.load_model to load model back as PyFuncModel and apply predict\n    prediction = mlflow.pyfunc.load_model("lightgbm_model").predict(df_trans.toPandas())\n    prediction = list(map(str, prediction))\n    mlflow.log_param("prediction", ",".join(prediction))\n\n    # Use mlflow.spark.load_model to load model back as PipelineModel and apply transform\n    predictions = mlflow.spark.load_model("lightgbm_model").transform(df_trans)\n    metrics = ComputeModelStatistics(evaluationMetric="classification", labelCol=\'labels\', scoredLabelsCol=\'prediction\').transform(predictions).collect()\n    mlflow.log_metric("accuracy", metrics[0][\'accuracy\'])\n')),(0,r.kt)("h2",{id:"cognitive-services"},"Cognitive Services"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'import mlflow\nfrom synapse.ml.cognitive import *\n\nwith mlflow.start_run():\n\n    text_key = "YOUR_COG_SERVICE_SUBSCRIPTION_KEY"\n    df = spark.createDataFrame([\n    ("I am so happy today, its sunny!", "en-US"),\n    ("I am frustrated by this rush hour traffic", "en-US"),\n    ("The cognitive services on spark aint bad", "en-US"),\n    ], ["text", "language"])\n\n    sentiment_model = (TextSentiment()\n                .setSubscriptionKey(text_key)\n                .setLocation("eastus")\n                .setTextCol("text")\n                .setOutputCol("prediction")\n                .setErrorCol("error")\n                .setLanguageCol("language"))\n\n    display(sentiment_model.transform(df))\n\n    mlflow.spark.save_model(sentiment_model, "sentiment_model")\n    mlflow.spark.log_model(sentiment_model, "sentiment_model")\n\n    output_df = mlflow.spark.load_model("sentiment_model").transform(df)\n    display(output_df)\n\n    # In order to call the predict function successfully you need to specify the\n    # outputCol name as `prediction`\n    prediction = mlflow.pyfunc.load_model("sentiment_model").predict(df.toPandas())\n    prediction = list(map(str, prediction))\n    mlflow.log_param("prediction", ",".join(prediction))\n')))}u.isMDXComponent=!0}}]);