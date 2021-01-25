//
//  File.swift
//  
//
//  Created by howard on 2021/1/25.
//
import Swiftensor
import Logging
import Foundation

class MnistLoader {
    static let logger = Logger(label: "ai.erurs.data.mnist.loader")
    static func loadMnist<T:Arithmetic>(path: String, type: T.Type = T.self) -> (train: (Tensor<T>, Tensor<Int32>), test: (Tensor<T>, Tensor<Int32>)) {
        do {
            let trainingData = try Data(contentsOf: URL(fileURLWithPath: path + "train-images.idx3-ubyte"))
            let trainingLabelData = try Data(contentsOf: URL(fileURLWithPath: path + "train-labels.idx1-ubyte"))
            let testingData = try Data(contentsOf: URL(fileURLWithPath: path + "t10k-images.idx3-ubyte"))
            let testingLabelData = try Data(contentsOf: URL(fileURLWithPath: path + "t10k-labels.idx1-ubyte"))
            
            let trainElements = trainingData.dropFirst(16).prefix(28 * 28 * 60_000).map(T.init)
            let testElements = testingData.dropFirst(16).prefix(28 * 28 * 60_000).map(T.init)
            
            let trainRawImages = Tensor<T>(trainElements)
            let testRawImages = Tensor<T>(testElements)
            
            let trainImages = trainRawImages / T(256)
            let testImages = testRawImages / T(256)
            
            let trainLabels = Tensor<Int32>(trainingLabelData.dropFirst(8).prefix(60_000).map(Int32.init))
            let testLabels = Tensor<Int32>(testingLabelData.dropFirst(8).prefix(10_000).map(Int32.init))
            
            return (
                train: (trainImages.reshaped([-1, 1, 28, 28]), trainLabels),
                test: (testImages.reshaped([-1, 1, 28, 28]), testLabels)
            )
        }catch let error {
            logger.error("error: \(error)")
            fatalError()
        }
    }
}
