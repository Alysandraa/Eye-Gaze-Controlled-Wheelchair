//H-bridge pins
const int EnA = 10;
const int In1 = 9;
const int In2 = 8;
const int In3 = 7;
const int In4 = 6;
const int EnB = 5;

float positionData;
float pos;

void setup() {
  Serial.begin(9600); 
  pinMode(EnA, OUTPUT);
  pinMode(In1, OUTPUT);
  pinMode(In2, OUTPUT);
  pinMode(In3, OUTPUT);
  pinMode(In4, OUTPUT);
  pinMode(EnB, OUTPUT);  
}

void loop() {
    //Serial.println("yes");
     positionData = Serial.parseFloat();
     //Serial.print("pos data");
     //Serial.println(positionData);
     pos = (positionData/360);
     //Serial.print("pos");
     //Serial.println(pos);
     fowards();
     direction();
}

void fowards(){
    digitalWrite(In1, LOW);
    digitalWrite(In2, HIGH);
    digitalWrite(In3, LOW);
    digitalWrite(In4, HIGH);
}

void direction(){
  if (pos <= 0.25 || pos >= 0.75){//steer right
   // Serial.println(position);
    int speedL = (pos * 255);
    String send_speedL = String(speedL);
    analogWrite(EnA, 255);
    analogWrite(EnB, (speedL));
    Serial.print("left motor speed");
    Serial.println(send_speedL);
  }else{//steer left
    int speedR = (pos * 255);
    String send_speedR = String(speedR);
    analogWrite(EnA, (speedR));
    analogWrite(EnB, 255);
    Serial.print("right motor speed");
    Serial.println(send_speedR);
  }
}
