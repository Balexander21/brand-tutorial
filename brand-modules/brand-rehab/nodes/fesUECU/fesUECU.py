"""
Establishes connection with the XPC Target and UECU,
takes decoded command inputs, converts them into 
stimulation parameters, and sends them to the UECU

@author: Ben Alexander
"""
import gc
import json
import logging
import time
from struct import pack
import scipy.io #used to import .mat pattern files
import socket #for UDP Connections
import array
import struct as st #For packing and unpacking the UECU package
import numpy as np
from brand import BRANDNode


class FesUECU(BRANDNode):

    def __init__(self):

        super().__init__()

        logging.info('The FES Node Started')
        # initialize stream info
        self.input_stream = self.parameters['input_stream'].encode()
        self.mouse_id = '$'
        #self.command_id = '$'

        # self.sync_key = self.parameters['sync_key'].encode()
        # self.time_key = self.parameters['time_key'].encode()

        # self.sync_dict = {}
        # self.sync_dict_json = json.dumps(self.sync_dict)
        self.i = 0
        self.patternStartingState = np.full(10,50,dtype=float) #Dummy value for testing
        self.patternsToEnable = np.concatenate([np.full(2, True), np.full(8, False)]) #Dummy value for testing
        self.patternStartingState = self.patternStartingState*self.patternsToEnable #set enabled patterns to starting state and all others to zero
        
        self.timePerSample = 0.02 #seconds
        self.numSecondsToSmooth = 2
        self.numSamplesToSmooth = self.numSecondsToSmooth/self.timePerSample

        self.firstLoop = True #When initializing node, denote that it is the first loop
        self.patternStates = self.patternStartingState #Set the current pattern states to predefined initial states (often 0% or 50%)
        self.patternsEnabled = np.full(10,False) #Sets all patterns to an unenabled state by default
        self.prevTime = None

        
        patternsUsedInds = [59,61] #The patterns to be used for each degree of freedom
        participantPatterns = scipy.io.loadmat("/home/ben/Documents/rp1_data.mat", mdict=None, appendmat=True) #Load in .mat file as a dictionary
        self.patternsUsed = [] #Initialize dictionary holding the patterns to be used for stimulation
        for patternInd in patternsUsedInds: #iterate of pattern indices 
            currentPattern = {} # initialize temporary dict for a given pattern
            for field in participantPatterns["Pattern"].dtype.names: #Iterate through pattern fields
                currentPattern.update({field: participantPatterns["Pattern"][field][0][patternInd]}) #create new dictionary of just the used pattern
            self.patternsUsed.append(currentPattern) #create list of pattern dicts
        
        #Setting up UDP ports for Host/Target/UECU communication
        targetIP = '192.168.30.1'
        hostIP = '192.168.30.3'
        targetPort = 52003
        self.targetAddress = (targetIP,targetPort)

    
    def run(self):
        while True:
            def multiInterp2(x, xp, fp):#Function for interpolating pattterns
                i = np.arange(x.size)
                j = np.searchsorted(xp, x) - 1
                d = (x - xp[j]) / (xp[j + 1] - xp[j])
                return (1 - d) * fp[i, j] + fp[i, j + 1] * d
            # read from cursor control stream
            reply = self.r.xread({self.input_stream: self.mouse_id},
                                 block=0,
                                 count=1)
            entries = reply[0][1]
            self.mouse_id, cursorFrame = entries[0]

            # pulling data in
            # sync_dict_in = json.loads(cursorFrame[self.sync_key].decode())
            # self.sync_dict = sync_dict_in

            sensors = np.frombuffer(cursorFrame[b'samples'],dtype=self.parameters['input_dtype']) #vel_x, vel_y = sensors
            sensors = np.concatenate((sensors,np.full(10-len(sensors),0)))
            sensors = sensors.astype(float)
            
            sensor_click = 0
            self.stateTime = time.monotonic() #Set current time
            if self.prevTime == None or self.firstLoop: #Determine if this is the first loop either through self.firstLoop or a lack of a previous time value
                self.prevTime = self.stateTime 
                self.loopNum = 0
                self.prevPacket = sensors #Velocity information for each pattern aka degree of freedom
                self.firstLoop = False #
            self.loopNum += 1 #tick up the loop number to count the number of loops
            self.deltaTime = self.stateTime - self.prevTime #determine the amount of time that has elapsed between the current and last loop
            if self.deltaTime > 0.2 or self.deltaTime < 0: #If deltaTime is too large or negative, set it to zero
                self.deltaTime = 0
                print("Adjusting dt")
            self.patternStates += 100*sensors*self.deltaTime #Convert velocity to position by multiplying by dt
            self.patternStates[self.patternStates<0] = 0 #Ensure the patternState doesn't go below pattern minimum
            self.patternStates[self.patternStates>100] = 100 #Ensure the patternState doesn't go above pattern maximum

            if np.any(np.abs(self.prevPacket - sensors) > 0.000001): #If the new packet is different than the previous one
                self.patternsEnabled = self.patternsEnabled | (abs(self.patternStates)>0.000001).astype(bool) #enable all patterns that were previously enabled or have a significant assigned pattern state
            
            self.prevPacket = sensors #Update prevPacket for next loop
            self.prevTime = self.stateTime #Update prevTime for next loop
            p = self.r.pipeline()
            
            ### Calculate and Pack Stim Parameters
            self.stimPackage = [0]*80 #PA*36 + PW*36 + Period*1 + Pct*1 + A*6
            self.stimPackage[72] = self.patternsUsed[0]["StimPeriod"][0][0] #Stimulation Period
            self.stimPackage[73] = self.patternStates[0] #Command Percentage
            self.pulse_widths = [0]*36
            self.pulse_amps = [0]*36
            for n in range(0,10):
                if self.patternsEnabled[n]:
                    pattern = self.patternsUsed[n]
                    pwArray = np.transpose(np.concatenate([pattern["Muscles"][0][0][5],pattern["Muscles"][0][1][5],pattern["Muscles"][0][2][5]],axis=1))
                    paArray = np.concatenate([pattern["Muscles"][0][0][0][0][0][1][0],pattern["Muscles"][0][1][0][0][0][1][0],pattern["Muscles"][0][2][0][0][0][1][0]])
                    breakpoints = pattern["BP"][0]
                    interp_point = np.ones(max(pwArray.shape))*self.patternStates[n]
                    self.pulse_widths += multiInterp2(interp_point, breakpoints, pwArray)
                    self.pulse_amps += paArray
            self.stimPackage[0:36] = self.pulse_amps
            self.stimPackage[36:72]=self.pulse_widths

            ### Send Stim Parameters to UECU
            packed_double = st.pack('d'*len(self.stimPackage),*self.stimPackage)
            unpacked_Uint8 = st.unpack('B'*len(self.stimPackage)*8,packed_double)
            stimPackage = unpacked_Uint8 

            sock = socket.socket(socket.AF_INET, # Internet 
                     socket.SOCK_DGRAM) # UDP
            if self.i % 100 == 0: #Sending UDP signals too frequently causes no signal to be recieved. This acts as a delay. Needs to be tuned
                sock.sendto(bytes(stimPackage), self.targetAddress) #Send 

            #logging.info('This part is working')
            p.execute()
            self.i += 1 #tick up the loop number by 1

"""
inputs: 
10D velocities

outputs:
FES Pulse Widths 36 
FES Amplitudes 36
FES Frequency 1
Command Percentage 1
6 Analog Channels? 6
Whole UECU package



%Expand muscle pattern states to channel pulse widths. 
pulseWidths = zeros(1,block.OutputPort(5).Dimensions);
for n=1:10
    if block.Dwork(9).Data(n)
        pulseWidths = pulseWidths + interp1(block.DialogPrm(n).Data(:,1), block.DialogPrm(n).Data(:, 2:(1+block.OutputPort(5).Dimensions) ), block.Dwork(8).Data(n), 'linear', 'extrap');
    end
    %^ first column is the break points, and the rest is the actual data
end


"""


if __name__ == "__main__":
    gc.disable()

    # setup
    fes_uecu = FesUECU()

    # main
    fes_uecu.run()

    gc.collect()
