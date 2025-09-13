---
status: on hold
priority: "10"
summary:
due to:
---

> GitHub Repo: https://github.com/lars-3010/Wallpaper-Generator
---

# main components
## main.ts
---
`main.ts` is the entry point of the Vue.js application. Its main responsibilities are:

1. Importing and setting up necessary Vue components and plugins.
2. Configuring Font Awesome icons.
3. Creating and mounting the Vue application.

Key points:
- It uses the `createApp` function from Vue to initialize the application.
- It imports and sets up Font Awesome icons using the `setupIcons` function.
- It globally registers Font Awesome components for use throughout the app.

## App.vue
---
`App.vue` is the root component of the application. It serves as the main container for the wallpaper generator. Its key features include:

1. Managing the state for cluster strength and number of clusters.
2. Rendering the WallpaperGenerator component.
3. Providing controls for adjusting wallpaper parameters.

Important aspects:
- It uses Vue's composition API with `setup` script.
- It uses computed properties for `clusterStrength` and `numClusters` to handle type conversions.
- It provides a method to rotate the wallpaper.

## WallpaperGenerator.vue
---
This component is responsible for generating and displaying the wallpaper. Its main functions are:

1. Creating clusters of symbols based on input parameters.
2. Placing symbols on the canvas.
3. Rendering the generated wallpaper.

Key points:
- It uses Vue's `defineComponent` for better TypeScript integration.
- It generates the wallpaper using the `SymbolPlacement` class and `createClusters` function.
- It provides methods for styling circles and icons.
- It includes a method to rotate the wallpaper.

# logic files
---
## symbolPlacement.ts
---
This file contains the `SymbolPlacement` class, which is responsible for placing symbols on the wallpaper. Its main functions include:

1. Placing symbols within clusters.
2. Placing random symbols across the canvas.
3. Creating and managing individual symbols.

Key features:
- It uses a `PlacedSymbol` interface to define the properties of a placed symbol.
- The `placeSymbolsInClusters` method distributes symbols within given clusters.
- The `placeRandomSymbols` method adds additional symbols randomly across the canvas.
- It includes a `gaussianRandom` method for creating more natural-looking distributions.

## wallpaperRotation.ts
---
This file provides functions for manipulating the wallpaper after it's generated. Its main responsibilities are:

1. Rotating the entire wallpaper.
2. Flipping the wallpaper horizontally or vertically.

Important aspects:
- The `rotateWallpaper` function applies a rotation transformation to all symbols.
- The `flipWallpaper` function can flip the wallpaper horizontally, vertically, or both.
- Both functions return a new array of symbols, preserving the immutability of the original data.

## clusterCreation.ts
---
This file is responsible for creating and managing clusters in the wallpaper. Its main functions are:

1. Creating an array of clusters based on the provided configuration.
2. Adjusting the intensity of existing clusters.

Key points:
- The `createClusters` function generates clusters with decreasing intensity.
- The `adjustClusterIntensity` function allows for post-creation adjustment of cluster intensities.
- Both functions return `ReadonlyArray<Cluster>` to ensure immutability of the returned data.

## symbolArrayConfig.ts
---
This file (renamed from arrayConfig.ts) manages the configuration for creating arrays of symbols. Its primary responsibilities include:

1. Defining interfaces for `SymbolPosition` and `SymbolArrayConfig`.
2. Providing a function to create a 2D array of symbols.

Important aspects:
- It uses `ReadonlyArray` to ensure immutability of the created symbol array.
- The `createSymbolArray` function generates a 2D array of symbols based on the provided configuration.
- It includes a `validateSymbolArrayConfig` function to check the validity of the configuration.

# config files
---
## symbolConfig.ts
---
This file manages the configuration for symbols (icons) used in the wallpaper. Its main responsibilities are:

1. Defining the `IconConfig` interface.
2. Providing a set of predefined icons with their configurations.
3. Offering utility functions for working with icons.

Key features:
- It defines an `IconConfig` interface with properties like icon, width, height, and angle.
- It provides a `ReadonlyArray` of predefined icon configurations.
- It includes functions to add icons to the Font Awesome library, get a random icon, and get a random symbol configuration.

## clusterConfig.ts
---
This file handles the configuration for clusters in the wallpaper. Its main functions are:

1. Defining interfaces for `ClusterConfig`, `Cluster`, and `IconPosition`.
2. Providing functions to create clusters and icon arrays.

Important aspects:
- It uses readonly properties in interfaces to ensure immutability.
- The `createClusters` function generates an array of clusters based on the provided configuration.
- The `createIconArray` function places icons within the clusters, respecting the cluster's intensity and the canvas boundaries.

## iconSetup.ts
---
This file is responsible for setting up and managing Font Awesome icons used in the application. Its primary functions are:

1. Importing specific icons from Font Awesome.
2. Providing a function to add these icons to the Font Awesome library.
3. Exporting an array of available icons for use in other parts of the application.

Key points:
- It centralizes icon management, making it easier to add or remove icons.
- The `setupIcons` function adds all imported icons to the Font Awesome library.
- The `availableIcons` array provides easy access to all available icons for use in symbol generation.
# unsortiert
---
**Ziel:** webbasiert, konfigurierbares Wallpaper bauen, das ausgewählte Symbole in Kreisen oder Hexagonen abbildet
> 	diese Formen sollen in Clustern nach außen hin weniger eingefärbt sein und zufällig verteilt sein
> 	bevor es gedreht wird, wird ein 2 dimensionales Array aus Symbolen gebaut -> mit konfiguriertem Abstand durchs Bild gehen und überall ein zufällig ausgewähltes Symbol hinsetzten
