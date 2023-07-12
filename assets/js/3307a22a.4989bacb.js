"use strict";(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[96975],{3905:function(e,t,n){n.d(t,{Zo:function(){return l},kt:function(){return f}});var r=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var c=r.createContext({}),u=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},l=function(e){var t=u(e.components);return r.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,i=e.originalType,c=e.parentName,l=s(e,["components","mdxType","originalType","parentName"]),d=u(n),f=o,m=d["".concat(c,".").concat(f)]||d[f]||p[f]||i;return n?r.createElement(m,a(a({ref:t},l),{},{components:n})):r.createElement(m,a({ref:t},l))}));function f(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=n.length,a=new Array(i);a[0]=d;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:o,a[1]=s;for(var u=2;u<i;u++)a[u]=n[u];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},1687:function(e,t,n){n.r(t),n.d(t,{assets:function(){return l},contentTitle:function(){return c},default:function(){return f},frontMatter:function(){return s},metadata:function(){return u},toc:function(){return p}});var r=n(83117),o=n(80102),i=(n(67294),n(3905)),a=["components"],s={title:"CognitiveServices - Create Audiobooks",hide_title:!0,status:"stable",name:"CognitiveServices - Create Audiobooks"},c="Create audiobooks using neural Text to speech",u={unversionedId:"features/cognitive_services/CognitiveServices - Create Audiobooks",id:"version-0.11.2/features/cognitive_services/CognitiveServices - Create Audiobooks",title:"CognitiveServices - Create Audiobooks",description:"Step 1: Load libraries and add service information",source:"@site/versioned_docs/version-0.11.2/features/cognitive_services/CognitiveServices - Create Audiobooks.md",sourceDirName:"features/cognitive_services",slug:"/features/cognitive_services/CognitiveServices - Create Audiobooks",permalink:"/SynapseML/docs/features/cognitive_services/CognitiveServices - Create Audiobooks",draft:!1,tags:[],version:"0.11.2",frontMatter:{title:"CognitiveServices - Create Audiobooks",hide_title:!0,status:"stable"},sidebar:"docs",previous:{title:"CognitiveServices - Create a Multilingual Search Engine from Forms",permalink:"/SynapseML/docs/features/cognitive_services/CognitiveServices - Create a Multilingual Search Engine from Forms"},next:{title:"CognitiveServices - Custom Search for Art",permalink:"/SynapseML/docs/features/cognitive_services/CognitiveServices - Custom Search for Art"}},l={},p=[{value:"Step 1: Load libraries and add service information",id:"step-1-load-libraries-and-add-service-information",level:2},{value:"Step 2: Attach the storage account to hold the audio files",id:"step-2-attach-the-storage-account-to-hold-the-audio-files",level:2},{value:"Step 3: Read in text data",id:"step-3-read-in-text-data",level:2},{value:"Step 4: Synthesize audio from text",id:"step-4-synthesize-audio-from-text",level:2},{value:"Step 5: Listen to an audio file",id:"step-5-listen-to-an-audio-file",level:2}],d={toc:p};function f(e){var t=e.components,n=(0,o.Z)(e,a);return(0,i.kt)("wrapper",(0,r.Z)({},d,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"create-audiobooks-using-neural-text-to-speech"},"Create audiobooks using neural Text to speech"),(0,i.kt)("h2",{id:"step-1-load-libraries-and-add-service-information"},"Step 1: Load libraries and add service information"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'from pyspark.sql import SparkSession\nfrom synapse.ml.core.platform import *\n\n# Bootstrap Spark Session\nspark = SparkSession.builder.getOrCreate()\nif running_on_synapse():\n    from notebookutils import mssparkutils\n    from notebookutils.visualization import display\n\n# Fill this in with your cognitive service information\nservice_key = find_secret(\n    "cognitive-api-key"\n)  # Replace this line with a string like service_key = "dddjnbdkw9329"\nservice_loc = "eastus"\n\nstorage_container = "audiobooks"\nstorage_key = find_secret("madtest-storage-key")\nstorage_account = "anomalydetectiontest"\n')),(0,i.kt)("h2",{id:"step-2-attach-the-storage-account-to-hold-the-audio-files"},"Step 2: Attach the storage account to hold the audio files"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'spark_key_setting = f"fs.azure.account.key.{storage_account}.blob.core.windows.net"\nspark.sparkContext._jsc.hadoopConfiguration().set(spark_key_setting, storage_key)\n')),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'import os\nfrom os.path import exists, join\n\nmount_path = f"wasbs://{storage_container}@{storage_account}.blob.core.windows.net/"\nif running_on_synapse():\n    mount_dir = join("/synfs", mssparkutils.env.getJobId(), storage_container)\n    if not exists(mount_dir):\n        mssparkutils.fs.mount(\n            mount_path, f"/{storage_container}", {"accountKey": storage_key}\n        )\nelif running_on_databricks():\n    if not exists(f"/dbfs/mnt/{storage_container}"):\n        dbutils.fs.mount(\n            source=mount_path,\n            mount_point=f"/mnt/{storage_container}",\n            extra_configs={spark_key_setting: storage_key},\n        )\n')),(0,i.kt)("h2",{id:"step-3-read-in-text-data"},"Step 3: Read in text data"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'from pyspark.sql.functions import udf\n\n\n@udf\ndef make_audio_filename(part):\n    return f"wasbs://{storage_container}@{storage_account}.blob.core.windows.net/alice_in_wonderland/part_{part}.wav"\n\n\ndf = (\n    spark.read.parquet(\n        "wasbs://publicwasb@mmlspark.blob.core.windows.net/alice_in_wonderland.parquet"\n    )\n    .repartition(10)\n    .withColumn("filename", make_audio_filename("part"))\n)\n\ndisplay(df)\n')),(0,i.kt)("h2",{id:"step-4-synthesize-audio-from-text"},"Step 4: Synthesize audio from text"),(0,i.kt)("div",null,(0,i.kt)("img",{src:"https://marhamilresearch4.blob.core.windows.net/gutenberg-public/Notebook/NeuralTTS_hero.jpeg",width:"500"})),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.cognitive import TextToSpeech\n\ntts = (\n    TextToSpeech()\n    .setSubscriptionKey(service_key)\n    .setTextCol("text")\n    .setLocation(service_loc)\n    .setErrorCol("error")\n    .setVoiceName("en-US-SteffanNeural")\n    .setOutputFileCol("filename")\n)\n\naudio = tts.transform(df).cache()\ndisplay(audio)\n')),(0,i.kt)("h2",{id:"step-5-listen-to-an-audio-file"},"Step 5: Listen to an audio file"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'from IPython.display import Audio\n\n\ndef get_audio_file(num):\n    if running_on_databricks():\n        return f"/dbfs/mnt/{storage_container}/alice_in_wonderland/part_{num}.wav"\n    else:\n        return join(mount_dir, f"alice_in_wonderland/part_{num}.wav")\n\n\nAudio(filename=get_audio_file(1))\n')))}f.isMDXComponent=!0}}]);