import { useState, useEffect, useCallback } from 'react';

export type DemoState = {
  personPosition: { x: number; y: number } | null;
  activeCamera: string | null;
  prediction: { start: {x:number, y:number}, end: {x:number, y:number}, active: boolean };
  currentStep: number;
};

export function useDemoMode() {
  const [demoState, setDemoState] = useState<DemoState>({
    personPosition: { x: 150, y: 150 }, // Near CAM-04
    activeCamera: 'CAM-04',
    prediction: { start: {x:0, y:0}, end: {x:0, y:0}, active: false },
    currentStep: 0
  });

  const runDemoStep = useCallback(() => {
    setDemoState(prev => {
      switch (prev.currentStep) {
        case 0:
          // P-102 moves to CAM-05
          return { ...prev, personPosition: { x: 300, y: 150 }, activeCamera: 'CAM-05', currentStep: 1 };
        case 1:
          // P-102 moves toward CAM-07 (Restricted zone)
          return { ...prev, personPosition: { x: 500, y: 200 }, activeCamera: 'CAM-07', currentStep: 2 };
        case 2:
          // P-102 enters blind spot between CAM-07 and CAM-09
          return { 
            ...prev, 
            personPosition: { x: 650, y: 250 }, 
            activeCamera: null, // Track lost
            currentStep: 3 
          };
        case 3:
          // System predicts route
          return { 
            ...prev,
            personPosition: { x: 650, y: 250 }, 
            prediction: { start: { x: 650, y: 250 }, end: { x: 800, y: 350 }, active: true },
            currentStep: 4
          };
        case 4:
          // CAM-09 detects P-102, Track reacquired
          return {
            ...prev,
            personPosition: { x: 800, y: 350 },
            activeCamera: 'CAM-09',
            prediction: { ...prev.prediction, active: false },
            currentStep: 0 // Reset demo
          };
        default:
          return prev;
      }
    });
  }, []);

  useEffect(() => {
    const timer = setInterval(runDemoStep, 4000); // Step every 4 seconds
    return () => clearInterval(timer);
  }, [runDemoStep]);

  return demoState;
}
