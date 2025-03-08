import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useState, useEffect } from "react";
import "./App.css";

function Courtroom() {
  return (
    <Canvas style={{ background: "lightgray", width: "100vw", height: "100vh" }} camera={{ position: [0, 5, 10], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <OrbitControls />
      <CourtroomLayout />
    </Canvas>
  );
}

function CourtroomLayout() {
  return (
    <>
      <TableWithLegs position={[-3, 0, 2]} color="blue" />
      <TableWithLegs position={[3, 0, 2]} color="red" />
      <JudgeTable position={[0, 0, -2]} />
      <HumanoidCharacter 
        position={[-3, 0, 2.5]} 
        suitColor="darkred" 
        tieColor="black"
        gestureType="presenting" 
        rotation={Math.PI} // Rotated 180 degrees to fully face the judge
      />
      <HumanoidCharacter 
        position={[3, 0, 2.5]} 
        suitColor="darkblue" 
        tieColor="black"
        gestureType="pointing" 
        rotation={Math.PI} // Rotated 180 degrees to fully face the judge
      />
      <JudgeCharacter position={[0, 0, -1.2]} />
    </>
  );
}

function TableWithLegs({ position, color }) {
  return (
    <>
      {/* Tabletop */}
      <mesh position={[position[0], position[1] + 1, position[2]]}>
        <boxGeometry args={[2, 0.2, 1]} />
        <meshStandardMaterial color={color} />
      </mesh>
      {/* Table Legs - Now Four Legs at Each Corner */}
      <mesh position={[position[0] - 0.9, position[1] + 0.5, position[2] - 0.4]}>
        <cylinderGeometry args={[0.1, 0.1, 1, 16]} />
        <meshStandardMaterial color="black" />
      </mesh>
      <mesh position={[position[0] + 0.9, position[1] + 0.5, position[2] - 0.4]}>
        <cylinderGeometry args={[0.1, 0.1, 1, 16]} />
        <meshStandardMaterial color="black" />
      </mesh>
      <mesh position={[position[0] - 0.9, position[1] + 0.5, position[2] + 0.4]}>
        <cylinderGeometry args={[0.1, 0.1, 1, 16]} />
        <meshStandardMaterial color="black" />
      </mesh>
      <mesh position={[position[0] + 0.9, position[1] + 0.5, position[2] + 0.4]}>
        <cylinderGeometry args={[0.1, 0.1, 1, 16]} />
        <meshStandardMaterial color="black" />
      </mesh>
    </>
  );
}

function JudgeCharacter({ position }) {
  return (
    <>
      {/* Head */}
      <mesh position={[position[0], position[1] + 2.8, position[2]-1.8]}>
        <boxGeometry args={[0.4, 0.4, 0.4]} />
        <meshStandardMaterial color="peachpuff" />
      </mesh>
      {/* Judge Robe - Body */}
      <mesh position={[position[0], position[1] + 2.2, position[2]-1.8]}>
        <boxGeometry args={[0.8, 0.8, 0.5]} />
        <meshStandardMaterial color="black" />
      </mesh>
      {/* Judge Collar */}
      <mesh position={[position[0], position[1] + 2.5, position[2] -1.7]}>
        <boxGeometry args={[0.6, 0.2, 0.1]} />
        <meshStandardMaterial color="white" />
      </mesh>
      
      {/* Left arm resting on bench */}
      <mesh position={[position[0] - 0.5, position[1] + 2.2, position[2]-1.8]}>
        <boxGeometry args={[0.2, 0.7, 0.2]} />
        <meshStandardMaterial color="black" />
      </mesh>
      
      {/* Right arm resting on bench */}
      <mesh position={[position[0] + 0.5, position[1] + 2.2, position[2]-1.8]}>
        <boxGeometry args={[0.2, 0.7, 0.2]} />
        <meshStandardMaterial color="black" />
      </mesh>
      {/* Lower Robe */}
      <mesh position={[position[0], position[1] + 1.5, position[2]-1.8]}>
        <boxGeometry args={[0.7, 0.6, 0.4]} />
        <meshStandardMaterial color="black" />
      </mesh>
    </>
  );
}

function HumanoidCharacter({ position, suitColor = "navy", tieColor = "black", gestureType = "none", rotation = 0 }) {
  const [armRotation, setArmRotation] = useState(0);
  
  // Animation for arm gestures
  useEffect(() => {
    let interval;
    if (gestureType !== "none") {
      interval = setInterval(() => {
        // Create a gentle oscillation for the arm
        const amplitude = gestureType === "presenting" ? 0.2 : 0.3;
        const center = gestureType === "presenting" ? -0.2 : -0.5;
        setArmRotation(center + Math.sin(Date.now() / 500) * amplitude);
      }, 50);
    }
    
    return () => clearInterval(interval);
  }, [gestureType]);

  return (
    <group rotation={[0, rotation, 0]}>
      {/* Head */}
      <mesh position={[position[0], position[1] + 1.8, position[2]-5.5]}>
        <boxGeometry args={[0.4, 0.4, 0.4]} />
        <meshStandardMaterial color="peachpuff" />
      </mesh>
      
      {/* Suit Jacket - Torso */}
      <mesh position={[position[0], position[1] + 1.2, position[2]-5.5]}>
        <boxGeometry args={[0.6, 0.8, 0.4]} />
        <meshStandardMaterial color={suitColor} />
      </mesh>
      
      {/* White Shirt */}
      <mesh position={[position[0], position[1] + 1.2, position[2]-5.29]}>
        <boxGeometry args={[0.4, 0.5, 0.05]} />
        <meshStandardMaterial color="white" />
      </mesh>
      
      {/* Shirt Collar */}
      <mesh position={[position[0], position[1] + 1.5, position[2]-5.27]}>
        <boxGeometry args={[0.4, 0.15, 0.05]} />
        <meshStandardMaterial color="white" />
      </mesh>
      
      {/* Tie - Made more prominent */}
      <mesh position={[position[0], position[1] + 1.3, position[2]-5.28]}>
        <boxGeometry args={[0.15, 0.4, 0.06]} />
        <meshStandardMaterial color={tieColor} />
      </mesh>
      <mesh position={[position[0], position[1] + 1.1, position[2]-5.28]}>
        <boxGeometry args={[0.1, 0.1, 0.06]} />
        <meshStandardMaterial color={tieColor} />
      </mesh>
      
      {/* Suit Jacket Lapels */}
      <mesh position={[position[0] - 0.2, position[1] + 1.4, position[2]-5.26]}>
        <boxGeometry args={[0.1, 0.3, 0.03]} />
        <meshStandardMaterial color={suitColor} />
      </mesh>
      <mesh position={[position[0] + 0.2, position[1] + 1.4, position[2]-5.26]}>
        <boxGeometry args={[0.1, 0.3, 0.03]} />
        <meshStandardMaterial color={suitColor} />
      </mesh>
      
      {/* Left Arm - This will be animated based on gesture */}
      <group 
        position={[position[0] - 0.42, position[1] + 1.2, position[2]-5.45]} 
        rotation={[0, 0, gestureType === "none" ? 0 : armRotation]}
      >
        {/* Upper Arm */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.2, 0.35, 0.2]} />
          <meshStandardMaterial color={suitColor} />
        </mesh>
        {/* Forearm - positioned relative to upper arm */}
        <mesh position={[0, -0.35, gestureType === "pointing" ? 0.15 : 0]}>
          <boxGeometry args={[0.18, 0.35, 0.18]} />
          <meshStandardMaterial color={suitColor} />
        </mesh>
        {/* Hand */}
        <mesh position={[0, -0.65, gestureType === "pointing" ? 0.25 : 0]}>
          <boxGeometry args={[0.15, 0.15, 0.15]} />
          <meshStandardMaterial color="peachpuff" />
        </mesh>
        {/* Pointing Finger (only for pointing gesture) */}
        {gestureType === "pointing" && (
          <mesh position={[0, -0.75, 0.35]}>
            <boxGeometry args={[0.05, 0.15, 0.05]} />
            <meshStandardMaterial color="peachpuff" />
          </mesh>
        )}
      </group>
      
      {/* Right Arm - Static or with different animation */}
      <group 
        position={[position[0] + 0.42, position[1] + 1.2, position[2]-5.5]}
        rotation={[0, 0, gestureType === "presenting" ? -armRotation : 0]}
      >
        {/* Upper Arm */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.2, 0.35, 0.2]} />
          <meshStandardMaterial color={suitColor} />
        </mesh>
        {/* Forearm */}
        <mesh position={[0, -0.35, 0]}>
          <boxGeometry args={[0.18, 0.35, 0.18]} />
          <meshStandardMaterial color={suitColor} />
        </mesh>
        {/* Hand */}
        <mesh position={[0, -0.65, 0]}>
          <boxGeometry args={[0.15, 0.15, 0.15]} />
          <meshStandardMaterial color="peachpuff" />
        </mesh>
      </group>
      
      {/* Suit Pants - Legs */}
      <mesh position={[position[0] - 0.2, position[1] + 0.4, position[2]-5.5]}>
        <boxGeometry args={[0.2, 0.8, 0.2]} />
        <meshStandardMaterial color={suitColor} />
      </mesh>
      <mesh position={[position[0] + 0.2, position[1] + 0.4, position[2]-5.5]}>
        <boxGeometry args={[0.2, 0.8, 0.2]} />
        <meshStandardMaterial color={suitColor} />
      </mesh>
      
      {/* Shoes */}
      <mesh position={[position[0] - 0.2, position[1] + 0.05, position[2]-5.4]}>
        <boxGeometry args={[0.22, 0.1, 0.4]} />
        <meshStandardMaterial color="black" />
      </mesh>
      <mesh position={[position[0] + 0.2, position[1] + 0.05, position[2]-5.4]}>
        <boxGeometry args={[0.22, 0.1, 0.4]} />
        <meshStandardMaterial color="black" />
      </mesh>
    </group>
  );
}

function JudgeTable({ position }) {
  return (
    <>
      {/* Table Base */}
      <mesh position={[position[0], position[1] + 1, position[2]]}>
        <boxGeometry args={[4, 2, 1]} />
        <meshStandardMaterial color="#E3B181" />
      </mesh>
      {/* Tabletop - Dark Brown */}
      <mesh position={[position[0], position[1] + 2.1, position[2]]}>
        <boxGeometry args={[4, 0.2, 1]} />
        <meshStandardMaterial color="#8B5A2B" />
      </mesh>
      {/* Side Panels */}
      <mesh position={[position[0] - 1.9, position[1] + 1, position[2] - 0.5]}>
        <boxGeometry args={[0.2, 2, 1]} />
        <meshStandardMaterial color="#E3B181" />
      </mesh>
      <mesh position={[position[0] + 1.9, position[1] + 1, position[2] - 0.5]}>
        <boxGeometry args={[0.2, 2, 1]} />
        <meshStandardMaterial color="#E3B181" />
      </mesh>
      {/* Gavel Base */}
      <mesh position={[position[0] + 0.5, position[1] + 2.20, position[2] - 0.3]}>
        <cylinderGeometry args={[0.2, 0.2, 0.05, 32]} />
        <meshStandardMaterial color="#E3B181" />
      </mesh>
      {/* Gavel Head */}
      <mesh position={[position[0] + 0.5, position[1] + 2.355, position[2] - 0.3]}>
        <cylinderGeometry args={[0.12, 0.12, 0.25, 16]} />
        <meshStandardMaterial color="black" />
      </mesh>
      {/* Gold Strips */}
      <mesh position={[position[0] + 0.5, position[1] + 2.255, position[2] - 0.3]}>
        <cylinderGeometry args={[0.13, 0.13, 0.03, 16]} />
        <meshStandardMaterial color="gold" />
      </mesh>
      <mesh position={[position[0] + 0.5, position[1] + 2.455, position[2] - 0.3]}>
        <cylinderGeometry args={[0.13, 0.13, 0.03, 16]} />
        <meshStandardMaterial color="gold" />
      </mesh>
      {/* Gavel Handle */}
      <mesh position={[position[0] + 0.71, position[1] + 2.35, position[2] - 0.3]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.035, 0.035, 0.6, 16]} />
        <meshStandardMaterial color="black" />
      </mesh>
    </>
  );
}

export default function CourtroomScene() {
  return (
    <div style={{ display: "flex" }}>
      <div style={{ width: "70vw", height: "100vh" }}>
        <Courtroom />
      </div>
    </div>
  );
}