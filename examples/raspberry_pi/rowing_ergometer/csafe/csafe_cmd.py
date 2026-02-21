#!/usr/bin/env python
#Copyright (c) 2011 Sam Gambrell, 2016-2017 Michael Droogleever
#Licensed under the Simplified BSD License.

# NOTE: This code has not been thoroughly tested and may not function as advertised.
# Please report and findings to the author so that they may be addressed in a stable release.

from warnings import warn
try:
    from . import csafe_dic  # Relative import for package use
except ImportError:
    import csafe_dic  # Direct import for script use

def __int2bytes(numbytes, integer):
    if not 0 <= integer <= 2 ** (8 * numbytes):
        raise ValueError("Integer is outside the allowable range")

    byte = []
    for k in range(numbytes):
        calcbyte = (integer >> (8 * k)) & 0xFF
        byte.append(calcbyte)

    return byte

def __bytes2int(raw_bytes):
    num_bytes = len(raw_bytes)
    integer = 0

    for k in range(num_bytes):
        integer = (raw_bytes[k] << (8 * k)) | integer

    return integer

def __bytes2ascii(raw_bytes):
    word = ""
    for letter in raw_bytes:
        word += chr(letter)

    return word

#for sending
def write(arguments):

    #priming variables
    i = 0
    message = []
    wrapper = 0
    wrapped = []
    maxresponse = 3 #start & stop flag & status

    #loop through all arguments
    while i < len(arguments):

        arg = arguments[i]
        cmdprop = csafe_dic.cmds[arg]
        command = []

        #load variables if command is a Long Command
        if len(cmdprop[1]) != 0:
            for varbytes in cmdprop[1]:
                i += 1
                intvalue = arguments[i]
                value = __int2bytes(varbytes, intvalue)
                command.extend(value)

            #data byte count
            cmdbytes = len(command)
            command.insert(0, cmdbytes)

        #add command id
        command.insert(0, cmdprop[0])

        #closes wrapper if required
        if len(wrapped) > 0 and (len(cmdprop) < 3 or cmdprop[2] != wrapper):
            wrapped.insert(0, len(wrapped)) #data byte count for wrapper
            wrapped.insert(0, wrapper) #wrapper command id
            message.extend(wrapped) #adds wrapper to message
            wrapped = []
            wrapper = 0

        #create or extend wrapper
        if len(cmdprop) == 3: #checks if command needs a wrapper
            if wrapper == cmdprop[2]: #checks if currently in the same wrapper
                wrapped.extend(command)
            else: #creating a new wrapper
                wrapped = command
                wrapper = cmdprop[2]
                maxresponse += 2

            command = [] #clear command to prevent it from getting into message

        #max message length
        cmdid = cmdprop[0] | (wrapper << 8)
        #account for response data + cmd echo byte + bytecount byte
        #add small margin for possible byte stuffing (unlikely for most data)
        resp_bytes = abs(sum(csafe_dic.resp[cmdid][1]))
        maxresponse += resp_bytes + 3  # data + cmd_echo + bytecount + margin

        #add completed command to final message
        message.extend(command)

        i += 1

    #closes wrapper if message ended on it
    if len(wrapped) > 0:
        wrapped.insert(0, len(wrapped)) #data byte count for wrapper
        wrapped.insert(0, wrapper) #wrapper command id
        message.extend(wrapped) #adds wrapper to message

    #prime variables
    checksum = 0x0
    j = 0

    #checksum and byte stuffing
    while j < len(message):
        #calculate checksum
        checksum = checksum ^ message[j]

        #byte stuffing
        if 0xF0 <= message[j] <= 0xF3:
            message.insert(j, csafe_dic.Byte_Stuffing_Flag)
            j += 1
            message[j] = message[j] & 0x3

        j += 1

    #add checksum to end of message
    message.append(checksum)

    #start & stop frames
    message.insert(0, csafe_dic.Standard_Frame_Start_Flag)
    message.append(csafe_dic.Stop_Frame_Flag)

    #check for frame size — report ID #2 allows up to 120 bytes of
    #CSAFE payload.  The frame itself (with start/stop/checksum) should
    #fit comfortably.
    if len(message) > 120:
        warn("Message is too long: " + str(len(message)))

    #Return the raw CSAFE frame — the transport layer (pyrow.py) handles
    #report ID selection and zero-padding for the HID report.
    return message


def __check_message(message):
    #prime variables
    i = 0
    checksum = 0

    #checksum and unstuff
    while i < len(message):
        #byte unstuffing
        if message[i] == csafe_dic.Byte_Stuffing_Flag:
            stuffvalue = message.pop(i + 1)
            message[i] = 0xF0 | stuffvalue

        #calculate checksum
        checksum = checksum ^ message[i]

        i = i + 1

    #checks checksum
    if checksum != 0:
        warn("Checksum error")
        return []

    #remove checksum from  end of message
    del message[-1]

    return message

#for recieving!!
def read(transmission):
    #prime variables
    message = []
    stopfound = False

    #reportid = transmission[0]
    startflag = transmission[1]

    if startflag == csafe_dic.Extended_Frame_Start_Flag:
        #destination = transmission[2]
        #source = transmission[3]
        j = 4
    elif startflag == csafe_dic.Standard_Frame_Start_Flag:
        j = 2
    else:
        warn("No Start Flag found.")
        return []

    while j < len(transmission):
        if transmission[j] == csafe_dic.Stop_Frame_Flag:
            stopfound = True
            break
        message.append(transmission[j])
        j += 1

    if not stopfound:
        if len(message) < 1:
            warn("No Stop Flag found and no data.")
            return []
        warn("No Stop Flag found — frame may be truncated.")

    message = __check_message(message)

    if not message:
        return []

    status = message.pop(0)

    #prime variables
    response = {'CSAFE_GETSTATUS_CMD' : [status,]}
    k = 0
    wrapend = -1
    wrapper = 0x0

    #loop through complete frames
    while k < len(message):
        result = []

        #bounds check – stop if we've run past available data (truncated frame)
        if k >= len(message):
            break

        #get command name
        msgcmd = message[k]
        if k <= wrapend:
            msgcmd = wrapper | msgcmd #check if still in wrapper
        if msgcmd not in csafe_dic.resp:
            break  # unknown command byte – likely hit padding/truncation
        msgprop = csafe_dic.resp[msgcmd]
        k = k + 1

        #get data byte count
        if k >= len(message):
            break
        bytecount = message[k]
        k = k + 1

        #if wrapper command then gets command in wrapper
        if msgprop[0] == 'CSAFE_SETUSERCFG1_CMD':
            wrapper = message[k - 2] << 8
            wrapend = k  + bytecount - 1
            if bytecount: #If wrapper length != 0
                if k >= len(message):
                    break
                msgcmd = wrapper | message[k]
                if msgcmd not in csafe_dic.resp:
                    break
                msgprop = csafe_dic.resp[msgcmd]
                k = k + 1
                if k >= len(message):
                    break
                bytecount = message[k]
                k = k + 1

        #special case for force plot and heartbeat data: variable-length response
        #The inner bytecount is always 33 (PM5 fixed template), but the FIRST
        #data byte (bytes_read) tells us how many of the following bytes are
        #real data.  The rest may be uninitialized PM5 memory (garbage).
        #Peek at bytes_read to build the correct response definition.
        if msgprop[0] in ('CSAFE_PM_GET_FORCEPLOTDATA', 'CSAFE_PM_GET_HEARTBEATDATA'):
            if bytecount > 0 and k < len(message):
                actual_data_bytes = message[k]  # peek at bytes_read
                num_samples = actual_data_bytes // 2
                msgprop = [msgprop[0], [1] + [2] * num_samples]  # copy, don't mutate global
            elif bytecount == 0:
                msgprop = [msgprop[0], [0,]]
            else:
                msgprop = [msgprop[0], [1]]  # just bytes_read, no samples available

        #special case for capability code, response lengths differ based off capability code
        if msgprop[0] == 'CSAFE_GETCAPS_CMD':
            msgprop = [msgprop[0], [1,] * bytecount]  # copy, don't mutate global

        #special case for get id, response length is variable
        if msgprop[0] == 'CSAFE_GETID_CMD':
            msgprop = [msgprop[0], [(-bytecount),]]  # copy, don't mutate global

        #checking that the recieved data byte is the expected length, sanity check
        if abs(sum(msgprop[1])) != 0 and bytecount != abs(sum(msgprop[1])):
            warn("Warning: bytecount is an unexpected length")

        #extract values
        for numbytes in msgprop[1]:
            n = abs(numbytes)
            raw_bytes = message[k:k + n]
            if len(raw_bytes) < n:
                raw_bytes = raw_bytes + [0] * (n - len(raw_bytes))
            value = (__bytes2int(raw_bytes) if numbytes >= 0 else __bytes2ascii(raw_bytes))
            result.append(value)
            k = k + n

        response[msgprop[0]] = result

    return response
