//
//  File.swift
//  
//
//  Created by howard on 2021/1/25.
//

import Foundation
import Logging
import Swiftensor

class DataLoader {
    static let logger = Logger(label: "ai.erurs.data.loader")

    static func loadBinary(path: String) -> Data {
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            return data
        } catch let error {
            logger.error("error: \(error)")
            fatalError()
        }
    }
}

