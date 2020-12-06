// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Eurus",
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "Eurus",
            targets: ["Eurus"]),
        .library(
            name: "Swiftensor",
            targets: ["Swiftensor"]
        ),
        .executable(name: "Examples", targets: ["Examples"])
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-log.git",from: "1.0.0")
    ],
    targets: [
        // Targets are the example building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "Eurus",
            dependencies: ["Swiftensor"],
            path: "./Sources/Eurus"
        ),
        .target(
            name: "Examples",
            dependencies: ["Eurus","Swiftensor"],
            path: "./Sources/Examples"
        ),
        .target(
            name: "Swiftensor",
            dependencies: [],
            path: "./Sources/Swiftensor"
        ),
        .testTarget(
            name: "EurusTests",
            dependencies: ["Eurus"]),
        .testTarget(
            name: "SwiftensorTests",
            dependencies: ["Swiftensor"]),
        
    ]
)
