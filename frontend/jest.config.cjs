/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jest-environment-jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  transform: {
    '^.+\\.[tj]sx?$': [
      'ts-jest',
      {
        // isolatedModules: true uses transpileModule() — skips semantic checks
        // so TypeScript won't throw on project-specific types like __VITE_BACKEND__
        isolatedModules: true,
        tsconfig: {
          jsx: 'react-jsx',
          module: 'CommonJS',
          moduleResolution: 'node',
          esModuleInterop: true,
          allowSyntheticDefaultImports: true,
          skipLibCheck: true,
        },
      },
    ],
  },
  // __VITE_BACKEND__ is replaced by Vite's define at build time.
  // In Jest we inject an empty string so resolveBase() falls through to window.location.
  globals: {
    __VITE_BACKEND__: '',
  },
  moduleNameMapper: {
    // Recharts ships both CJS and ESM — Jest must use the CJS build
    '^recharts$': '<rootDir>/node_modules/recharts/lib/index.js',
  },
};
