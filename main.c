#include <stdio.h>


// int main(void) {
//   for (int i = 1; i <= 100; i++)
//   {
//     for (int j = 1; j <= 9; j++)
//       {
//         printf("%d x %d = %d\n", i, j, (i * j));
//       }
//     printf("\n");
//
//   }
//   # Copyright 2021 The gRPC Authors
// #
// # Licensed under the Apache License, Version 2.0 (the "License");
// # you may not use this file except in compliance with the License.
// # You may obtain a copy of the License at
// #
// #     http://www.apache.org/licenses/LICENSE-2.0
// #
// # Unless required by applicable law or agreed to in writing, software
// # distributed under the License is distributed on an "AS IS" BASIS,
// # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// # See the License for the specific language governing permissions and
// # limitations under the License.
// """Generates and compiles C++ grpc stubs from proto_library rules."""
//
// load("@rules_proto//proto:defs.bzl", "proto_library")
// load("//bazel:generate_cc.bzl", "generate_cc")
//load("//bazel:protobuf.bzl", "well_known_proto_libs")

// def cc_grpc_library(
//         name,
//         srcs,
//         deps,
//         proto_only = False,
//         well_known_protos = False,
//         generate_mocks = False,
//         use_external = False,
//         grpc_only = False,
//         **kwargs):
//     """Generates C++ grpc classes for services defined in a proto file.
//
//     If grpc_only is True, this rule is compatible with proto_library and
//     cc_proto_library native rules such that it expects proto_library target
//     as srcs argument and generates only grpc library classes, expecting
//     protobuf messages classes library (cc_proto_library target) to be passed in
//     deps argument. By default grpc_only is False which makes this rule to behave
//     in a backwards-compatible mode (trying to generate both proto and grpc
//     classes).
//
//     Assumes the generated classes will be used in cc_api_version = 2.
//
//     Args:
//         name (str): Name of rule.
//         srcs (list): A single .proto file which contains services definitions,
//           or if grpc_only parameter is True, a single proto_library which
//           contains services descriptors.
//         deps (list): A list of C++ proto_library (or cc_proto_library) which
//           provides the compiled code of any message that the services depend on.
//         proto_only (bool): If True, create only C++ proto classes library,
//           avoid creating C++ grpc classes library (expect it in deps).
//           Deprecated, use native cc_proto_library instead. False by default.
//         well_known_protos (bool): Should this library additionally depend on
//           well known protos. Deprecated, the well known protos should be
//           specified as explicit dependencies of the proto_library target
//           (passed in srcs parameter) instead. False by default.
//         generate_mocks (bool): when True, Google Mock code for client stub is
//           generated. False by default.
//         use_external (bool): Not used.
//         grpc_only (bool): if True, generate only grpc library, expecting
//           protobuf messages library (cc_proto_library target) to be passed as
//           deps. False by default (will become True by default eventually).
//         **kwargs: rest of arguments, e.g., compatible_with and visibility
//     """
//     if len(srcs) > 1:
//         fail("Only one srcs value supported", "srcs")
//     if grpc_only and proto_only:
//         fail("A mutualy exclusive configuration is specified: grpc_only = True and proto_only = True")
//
//     extra_deps = []
//     proto_targets = []
//
//     if not grpc_only:
//         proto_target = "_" + name + "_only"
//         cc_proto_target = name if proto_only else "_" + name + "_cc_proto"
//
//         proto_deps = ["_" + dep + "_only" for dep in deps if dep.find(":") == -1]
//         proto_deps += [dep.split(":")[0] + ":" + "_" + dep.split(":")[1] + "_only" for dep in deps if dep.find(":") != -1 and dep.find("com_google_googleapis") == -1]
//         proto_deps += [dep for dep in deps if dep.find("com_google_googleapis") != -1]
//         if well_known_protos:
//             proto_deps += well_known_proto_libs()
//         proto_library(
//             name = proto_target,
//             srcs = srcs,
//             deps = proto_deps,
//             **kwargs
//         )
//
//         native.cc_proto_library(
//             name = cc_proto_target,
//             deps = [":" + proto_target],
//             **kwargs
//         )
//         extra_deps.append(":" + cc_proto_target)
//         proto_targets.append(proto_target)
//     else:
//         if not srcs:
//             fail("srcs cannot be empty", "srcs")
//         proto_targets += srcs
//
//     if not proto_only:
//         codegen_grpc_target = "_" + name + "_grpc_codegen"
//         generate_cc(
//             name = codegen_grpc_target,
//             srcs = proto_targets,
//             plugin = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
//             well_known_protos = well_known_protos,
//             generate_mocks = generate_mocks,
//             **kwargs
//         )

//         native.cc_library(
//             name = name,
//             srcs = [":" + codegen_grpc_target],
//             hdrs = [":" + codegen_grpc_target],
//             deps = deps +
//                    extra_deps +
//                    ["@com_github_grpc_grpc//:grpc++_codegen_proto"],
//             **kwargs
//         )
//
//         """
// Convolutional Neural Network
//
// Objective : To train a CNN model detect if TB is present in Lung X-ray or not.
//
// Resources CNN Theory :
//     https://en.wikipedia.org/wiki/Convolutional_neural_network
// Resources Tensorflow : https://www.tensorflow.org/tutorials/images/cnn
//
// Download dataset from :
// https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html
//
// 1. Download the dataset folder and create two folder training set and test set
// in the parent dataset folder
// 2. Move 30-40 image from both TB positive and TB Negative folder
// in the test set folder
// 3. The labels of the images will be extracted from the folder name
// the image is present in.
//
// """
//
// # Part 1 - Building the CNN
//
// import numpy as np
//
// # Importing the Keras libraries and packages
// import tensorflow as tf
// from keras import layers, models
//
// if __name__ == "__main__":
//     # Initialising the CNN
//     # (Sequential- Building the model layer by layer)
//     classifier = models.Sequential()
//
//     # Step 1 - Convolution
//     # Here 64,64 is the length & breadth of dataset images and 3 is for the RGB channel
//     # (3,3) is the kernel size (filter matrix)
//     classifier.add(
//         layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu")
//     )
//
//     # Step 2 - Pooling
//     classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))
//
//     # Adding a second convolutional layer
//     classifier.add(layers.Conv2D(32, (3, 3), activation="relu"))
//     classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))
//
//     # Step 3 - Flattening
//     classifier.add(layers.Flatten())
//
//     # Step 4 - Full connection
//     classifier.add(layers.Dense(units=128, activation="relu"))
//     classifier.add(layers.Dense(units=1, activation="sigmoid"))
//
//     # Compiling the CNN
//     classifier.compile(
//         optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
//     )
//
//     # Part 2 - Fitting the CNN to the images
//
//     # Load Trained model weights
//
//     # from keras.models import load_model
//     # regressor=load_model('cnn.h5')
//
//     train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
//         rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
//     )
//
//     test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
//
//     training_set = train_datagen.flow_from_directory(
//         "dataset/training_set", target_size=(64, 64), batch_size=32, class_mode="binary"
//     )
//
//     test_set = test_datagen.flow_from_directory(
//         "dataset/test_set", target_size=(64, 64), batch_size=32, class_mode="binary"
//     )
//
//     classifier.fit_generator(
//         training_set, steps_per_epoch=5, epochs=30, validation_data=test_set
//     )
//
//     classifier.save("cnn.h5")
//
//     # Part 3 - Making new predictions
//
//     test_image = tf.keras.preprocessing.image.load_img(
//         "dataset/single_prediction/image.png", target_size=(64, 64)
//     )
//     test_image = tf.keras.preprocessing.image.img_to_array(test_image)
//     test_image = np.expand_dims(test_image, axis=0)
//     result = classifier.predict(test_image)
//     # training_set.class_indices
//     if result[0][0] == 0:
//         prediction = "Normal"
//     if result[0][0] == 1:
//         prediction = "Abnormality detected"
//
// #Lists (Ro'yxatlar)


// mevalar = ['olma','Anjir','bexi','Gilos','Siliva','shaftoli',"o'rik"]
// narxlar = [12000, 18000, 10900, 22000]
// sonlar = ['bir', 'ikki', 35, 46, 58]
// mevalar[3] = 'banan'
// print(mevalar[3])
// mevalar.append('Sham')# ro'yxatni faqat oxiriga qo'shadi
// mevalar.insert(0, 'yabloko') #ro'yxatni index orqali xoxlagan joyiga qo'shish mumkin
// print(mevalar)
//
// cars = []
// cars.append('lasetti')
// cars.append('malibu')
// cars.append('tracker')
// del cars[0] #objectni olib tashlash
// cars.insert(0, 'nexia 3')
// print(cars)
// hayvonlar = ['it', 'mushuk', 'sigir', 'quyon', 'qo\'y', 'mushuk']
// hayvonlar.remove('mushuk')# faqat boshidagi e'lementni olib tashlaydi
// print(hayvonlar)
//
// bozorlik = ['un', 'banan', 'piyoz']
// #oldim = bozorlik.pop(2)
//
// #print('Men ' + oldim + ' sotib oldim' )
// #print('olinmagan maxsulotlar:',  bozorlik)
//
// maxsulot2 = bozorlik.pop()
// print(maxsulot2)


  return 0;
}
/*
 * @author: jelathro
 * @date: 11/6/13
 */

#include <stdlib.h>

#include "CircularBuffer.h"

CircularBuffer * circularbuffer_initialize(size_t size, void * val){
	size_t i;
	CircularBuffer * cb = (CircularBuffer *)malloc( sizeof(CircularBuffer) );

        if (cb == NULL) {

            // any other implementation may be added here.

            printf("\nERROR: Insufficient memory. Terminating...");
            exit(EXIT_FAILURE);
        }

	cb->buffer = (Item *)calloc(size, sizeof(Item));

        if (cb->buffer == NULL) {

            // any other implementation may be added here.

            printf("\nERROR: Insufficient memory. Terminating...");
            exit(EXIT_FAILURE);
        }

	for(i=0; i<size; i++){
		cb->buffer[i].data = val;
	}

	cb->size = size;
	cb->start = 0;
	cb->end = 0;

	return(cb);
}

int circularbuffer_add(CircularBuffer * cb, void * val){
	cb->buffer[ cb->end ].data = val;
	cb->end = (cb->end + 1) % cb->size;

	if( cb->end == cb->start){
		cb->start = (cb->start + 1) % cb->size;
	}

	return(1);
}

void * circularbuffer_read(CircularBuffer * cb){
	size_t start = cb->start;
	cb->start = (cb->start + 1) % cb->size;

	return( cb->buffer[ start ].data );
}

int circularbuffer_destroy(CircularBuffer * cb, circularbuffer_destroybuffer df){
	size_t i;

	for(i=0; i<cb->size; i++){
		df( cb->buffer[i].data );
	}

	free(cb->buffer);
	return(1);
}